"""
HelmetClassifierNet v5 — Upgraded CNN Classifier

Değişiklikler (v4 → v5):
  ✦ Focal Loss + Label Smoothing → overconfidence azaltma
  ✦ Mixup / CutMix augmentation → decision boundary düzleştirme
  ✦ EdgeTextureBranch → renk yerine yüzey dokusu/kenar bilgisi
  ✦ 384×384 input desteği → ince detayları yakalama
  ✦ Test-Time Augmentation (TTA) → inference robustness
  ✦ Grad-CAM hook desteği → model debugging
  ✦ Center-guided spatial prior → crop merkezindeki baş/helmet bölgesine odak

Mimari:
  EfficientNet-B0 backbone + CBAM + Center-Guided FPN + EdgeTexture Branch
  (128 FPN + 64 edge) = 192 → 128 → 64 → 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.ops import deform_conv2d
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from src.utils.image_ops import AspectRatioPadResize


DEFAULT_CENTER_GUIDANCE_SIGMA_X = 0.50
DEFAULT_CENTER_GUIDANCE_SIGMA_Y = 0.42
DEFAULT_CENTER_GUIDANCE_CENTER_X = 0.0
DEFAULT_CENTER_GUIDANCE_CENTER_Y = -0.18

DEFAULT_HELMET_FOCUS_LOSS_WEIGHT = 0.10
DEFAULT_HELMET_FOCUS_SIGMA_X = 0.40
DEFAULT_HELMET_FOCUS_SIGMA_Y = 0.24
DEFAULT_HELMET_FOCUS_CENTER_X = 0.0
DEFAULT_HELMET_FOCUS_CENTER_Y = -0.30
DEFAULT_HELMET_FOCUS_OUTSIDE_PENALTY = 0.15


def resolve_guidance_config(
    legacy_sigma=None,
    sigma_x=None,
    sigma_y=None,
    center_x=None,
    center_y=None,
    *,
    default_sigma_x=DEFAULT_CENTER_GUIDANCE_SIGMA_X,
    default_sigma_y=DEFAULT_CENTER_GUIDANCE_SIGMA_Y,
    default_center_x=DEFAULT_CENTER_GUIDANCE_CENTER_X,
    default_center_y=DEFAULT_CENTER_GUIDANCE_CENTER_Y,
):
    """Resolve legacy single-sigma config into explicit anisotropic prior values."""
    if sigma_x is None:
        sigma_x = legacy_sigma if legacy_sigma is not None else default_sigma_x
    if sigma_y is None:
        sigma_y = legacy_sigma if legacy_sigma is not None else default_sigma_y
    if center_x is None:
        center_x = default_center_x
    if center_y is None:
        center_y = default_center_y
    return float(sigma_x), float(sigma_y), float(center_x), float(center_y)


def legacy_guidance_sigma(sigma_x, sigma_y):
    """Persist a backwards-compatible single sigma for older tooling."""
    return float((float(sigma_x) + float(sigma_y)) / 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss — zor örneklere daha fazla ağırlık verir.

    Kolay örneklerin (model zaten %99 doğru biliyor) gradient'ini bastırır,
    borderline vakalara (koyu kask ↔ koyu saç) odaklanmayı zorlar.

    Args:
        alpha:  Class balance ağırlığı (0-1). 0.25 = minority class boost.
        gamma:  Focusing parametresi (0-5). Yüksek → zor örneklere daha çok odak.
                gamma=0 → standart CE, gamma=2 → iyi başlangıç.
        label_smoothing: Soft target parametresi (0-0.2). Overconfidence azaltır.
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# MIXUP / CUTMIX AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.4):
    """
    Mixup: İki görüntüyü lambda oranında karıştırır.
    Decision boundary'yi düzleştirir, overconfidence azaltır.

    Returns:
        mixed_x:  Karışık görüntü batch'i
        y_a, y_b: Orijinal label'lar
        lam:      Karışım oranı
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: Bir görüntünün bir bölgesini başka bir görüntüyle değiştirir.
    Mixup'tan daha agresif — model yerel özelliklere odaklanmaya zorlanır.

    Returns:
        mixed_x:  CutMix uygulanmış batch
        y_a, y_b: Orijinal label'lar
        lam:      Orijinal alan oranı (kesilen alan sonrası)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Gerçek lambda'yı hesapla (clip sonrası alan değişebilir)
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup / CutMix için weighted loss hesaplama."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class HelmetDataset(Dataset):
    """Helmet/No-Helmet dataset için PyTorch Dataset."""
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []

        helmet_dir = self.data_dir / 'helmet'
        no_helmet_dir = self.data_dir / 'no_helmet'

        for ext in ['*.png', '*.jpg']:
            for img_path in helmet_dir.glob(ext):
                self.samples.append(img_path)
                self.labels.append(1)
            for img_path in no_helmet_dir.glob(ext):
                self.samples.append(img_path)
                self.labels.append(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MODULES
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """CBAM Channel Attention — hangi feature map'ler önemli."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attention = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        return x * attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention — görüntünün neresine bakmalı."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module = Channel + Spatial."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


def build_center_prior(
    height,
    width,
    device,
    dtype,
    sigma_x=0.6,
    sigma_y=0.6,
    center_x=0.0,
    center_y=0.0,
):
    """Create a 2D Gaussian prior that softly emphasizes a target crop region."""
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype).view(1, 1, 1, width)
    sigma_x = max(float(sigma_x), 1e-3)
    sigma_y = max(float(sigma_y), 1e-3)
    prior = torch.exp(
        -0.5
        * (
            (((xs - float(center_x)) / sigma_x) ** 2)
            + (((ys - float(center_y)) / sigma_y) ** 2)
        )
    )
    return prior


class CenterPriorSpatialGate(nn.Module):
    """
    Lightweight guided attention that softly amplifies the crop center.

    This is intentionally parameter-free so older checkpoints stay compatible.
    """
    def __init__(self, strength=0.35, sigma_x=0.6, sigma_y=0.6, center_x=0.0, center_y=0.0):
        super().__init__()
        self.strength = float(strength)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.center_x = float(center_x)
        self.center_y = float(center_y)

    def forward(self, x):
        prior = build_center_prior(
            x.shape[2],
            x.shape[3],
            x.device,
            x.dtype,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            center_x=self.center_x,
            center_y=self.center_y,
        )
        prior = prior / prior.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gain = 1.0 + self.strength * (prior - 1.0)
        return x * gain


# ═══════════════════════════════════════════════════════════════════════════════
# DEFORMABLE CONVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

class DeformableConv2d(nn.Module):
    """Deformable Convolution v2 — geometrik dönüşümlere uyum."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.5)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(x, offset, self.weight, bias=self.bias,
                             padding=self.padding, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE FEATURE POOLING
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFeaturePooling(nn.Module):
    """FPN ölçeklerine attention-based weighted fusion."""
    def __init__(self, feature_dim=128, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        concat = stacked.view(stacked.size(0), -1)
        weights = self.attention(concat).unsqueeze(2)
        return (stacked * weights).sum(dim=1)


class CenterWeightedPooling(nn.Module):
    """Pool features with a soft center prior instead of flat global averaging."""
    def __init__(self, sigma_x=0.6, sigma_y=0.6, center_x=0.0, center_y=0.0):
        super().__init__()
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.center_x = float(center_x)
        self.center_y = float(center_y)

    def forward(self, x):
        weights = build_center_prior(
            x.shape[2],
            x.shape[3],
            x.device,
            x.dtype,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            center_x=self.center_x,
            center_y=self.center_y,
        )
        weights = weights / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x * weights).sum(dim=(2, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE TEXTURE BRANCH  (v5 — replaces color_branch)
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeTextureBranch(nn.Module):
    """
    Renk yerine yüzey dokusu & kenar bilgisi çıkarır.

    Sorun: color_branch raw RGB'den öğreniyor → koyu kask ≈ koyu saç.
    Çözüm: Sabit Sobel filtresi ile kenar haritası çıkar,
           grayscale + edge_x + edge_y → 3 kanallı yapısal girdi.
           Bu sayede model renk yerine doku ve kenar farkını öğrenir.
           (Kask: pürüzsüz yüzey, keskin kenarlar / Saç: dokulu, dağınık kenarlar)

    Input:  (B, 3, H, W) RGB tensor (normalleştirilmiş)
    Output: (B, 64) edge-texture feature vector
    """
    def __init__(self, out_dim=64):
        super().__init__()

        # Sabit Sobel kernelleri (öğrenilmez, register_buffer ile)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # ImageNet normalizasyonunu geri almak için parametreler
        self.register_buffer('denorm_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('denorm_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 3 kanal girdi: grayscale + edge_x + edge_y
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        # Denormalize → gerçek piksel değerleri (0-1)
        x_raw = x * self.denorm_std + self.denorm_mean

        # RGB → Grayscale
        gray = 0.299 * x_raw[:, 0:1] + 0.587 * x_raw[:, 1:2] + 0.114 * x_raw[:, 2:3]

        # Sobel kenar haritaları
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)

        # [grayscale, edge_x, edge_y] → yapısal 3-kanal girdi
        structural = torch.cat([gray, edge_x, edge_y], dim=1)

        return self.features(structural).view(x.size(0), -1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HelmetClassifierNet(nn.Module):
    """
    HelmetClassifierNet v5.

    EfficientNet-B0 backbone + CBAM + FPN + EdgeTexture Branch.

    Değişiklikler (v4 → v5):
      - color_branch → EdgeTextureBranch (kenar + doku, renk bağımsız)
      - branch_type parametresi: 'edge_texture' (yeni default) veya 'color' (legacy)
      - Grad-CAM hook noktası: self.fpn_output (register ile)

    Mimari:
      Backbone:    EfficientNet-B0 features, 3 ölçek (40ch, 112ch, 1280ch)
      Attention:   Her ölçekte CBAM + merkez öncelikli spatial guidance
      FPN:         Top-down lateral, tüm ölçekler 128-dim
      Side Branch: EdgeTexture (grayscale + Sobel → 64-dim)
      Pooling:     Center-weighted pooling + scale fusion
      Classifier:  (128 + 64) = 192 → 128 → 64 → 2

    Giriş: 224×224 veya 384×384 RGB
    """
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        branch_type='edge_texture',
        center_guidance=False,
        center_guidance_strength=0.35,
        center_guidance_sigma_x=DEFAULT_CENTER_GUIDANCE_SIGMA_X,
        center_guidance_sigma_y=DEFAULT_CENTER_GUIDANCE_SIGMA_Y,
        center_guidance_center_x=DEFAULT_CENTER_GUIDANCE_CENTER_X,
        center_guidance_center_y=DEFAULT_CENTER_GUIDANCE_CENTER_Y,
    ):
        super().__init__()
        self.branch_type = branch_type
        self.center_guidance = center_guidance

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.stage1 = backbone.features[:4]   # → 40ch
        self.stage2 = backbone.features[4:6]  # → 112ch
        self.stage3 = backbone.features[6:]   # → 1280ch

        self.cbam1 = CBAM(40, reduction=8)
        self.cbam2 = CBAM(112, reduction=16)
        self.cbam3 = CBAM(1280, reduction=16)

        self.lateral3 = nn.Conv2d(1280, 128, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(112, 128, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(40, 128, kernel_size=1, bias=False)

        self.smooth2 = DeformableConv2d(128, 128, kernel_size=3, padding=1)
        self.smooth1 = DeformableConv2d(128, 128, kernel_size=3, padding=1)

        self.center_guide_p5 = CenterPriorSpatialGate(
            strength=center_guidance_strength,
            sigma_x=center_guidance_sigma_x,
            sigma_y=center_guidance_sigma_y,
            center_x=center_guidance_center_x,
            center_y=center_guidance_center_y,
        )
        self.center_guide_p4 = CenterPriorSpatialGate(
            strength=center_guidance_strength,
            sigma_x=center_guidance_sigma_x,
            sigma_y=center_guidance_sigma_y,
            center_x=center_guidance_center_x,
            center_y=center_guidance_center_y,
        )
        self.center_guide_p3 = CenterPriorSpatialGate(
            strength=center_guidance_strength,
            sigma_x=center_guidance_sigma_x,
            sigma_y=center_guidance_sigma_y,
            center_x=center_guidance_center_x,
            center_y=center_guidance_center_y,
        )
        self.fpn_cbam3 = CBAM(128, reduction=8)
        self.fpn_cbam2 = CBAM(128, reduction=8)
        self.fpn_cbam1 = CBAM(128, reduction=8)

        # Side branch: edge_texture (v5) veya color (v4 legacy)
        if branch_type == 'edge_texture':
            self.side_branch = EdgeTextureBranch(out_dim=64)
        else:
            # Legacy color branch (v4 uyumluluğu)
            self.side_branch = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )

        self.adaptive_pooling = AdaptiveFeaturePooling(feature_dim=128, num_scales=3)
        self.center_pool = CenterWeightedPooling(
            sigma_x=center_guidance_sigma_x,
            sigma_y=center_guidance_sigma_y,
            center_x=center_guidance_center_x,
            center_y=center_guidance_center_y,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # ── Grad-CAM hook noktası ────────────────────────────────────────
        self._gradcam_activations = None
        self._gradcam_gradients = None

    def _save_activation(self, module, input, output):
        self._gradcam_activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradcam_gradients = grad_output[0]

    def register_gradcam_hooks(self, target_layer=None):
        """
        Grad-CAM hook'larını kaydeder.
        Args:
            target_layer: Hook uygulanacak katman. None ise fpn_cbam3 kullanılır.
        Returns:
            hooks: Temizlemek için hook handle listesi.
        """
        if target_layer is None:
            target_layer = self.fpn_cbam3

        h1 = target_layer.register_forward_hook(self._save_activation)
        h2 = target_layer.register_full_backward_hook(self._save_gradient)
        return [h1, h2]

    def freeze_backbone(self):
        for stage in [self.stage1, self.stage2, self.stage3]:
            for param in stage.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for stage in [self.stage1, self.stage2, self.stage3]:
            for param in stage.parameters():
                param.requires_grad = True

    def forward(self, x, return_aux=False):
        # Side branch (edge-texture veya color)
        if self.branch_type == 'edge_texture':
            side_feat = self.side_branch(x)
        else:
            side_feat = self.side_branch(x).view(x.size(0), -1)

        # Backbone + CBAM
        c3 = self.cbam1(self.stage1(x))
        c4 = self.cbam2(self.stage2(c3))
        c5 = self.cbam3(self.stage3(c4))

        # FPN
        p5 = self.lateral3(c5)
        p4 = self.smooth2(self.lateral2(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest'))
        p3 = self.smooth1(self.lateral1(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest'))

        if self.center_guidance:
            # Post-FPN attention with a soft center prior.
            p5 = self.fpn_cbam3(self.center_guide_p5(p5))
            p4 = self.fpn_cbam2(self.center_guide_p4(p4))
            p3 = self.fpn_cbam1(self.center_guide_p3(p3))

            # Center-weighted pooling suppresses edge/background clutter inside the crop.
            p5_pool = self.center_pool(p5)
            p4_pool = self.center_pool(p4)
            p3_pool = self.center_pool(p3)
        else:
            p5 = self.fpn_cbam3(p5)
            p4 = self.fpn_cbam2(p4)
            p3 = self.fpn_cbam1(p3)

            p5_pool = F.adaptive_avg_pool2d(p5, 1).view(x.size(0), -1)
            p4_pool = F.adaptive_avg_pool2d(p4, 1).view(x.size(0), -1)
            p3_pool = F.adaptive_avg_pool2d(p3, 1).view(x.size(0), -1)

        fpn_feat = self.adaptive_pooling([p3_pool, p4_pool, p5_pool])

        # Fuse + classify
        fused = torch.cat([fpn_feat, side_feat], dim=1)
        logits = self.classifier(fused)

        if not return_aux:
            return logits

        aux = {
            'fpn_maps': {
                'p3': p3,
                'p4': p4,
                'p5': p5,
            },
            'fpn_pooled': {
                'p3': p3_pool,
                'p4': p4_pool,
                'p5': p5_pool,
            },
            'fpn_fused': fpn_feat,
            'side_feat': side_feat,
        }
        return logits, aux


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CNNClassifier:
    """
    CNN-based helmet classifier v5.

    Yeni özellikler:
      - loss_type:       'focal' (default) veya 'ce'
      - label_smoothing: 0.1 (default)
      - mixup_alpha:     0.4 (default, 0 = kapalı)
      - cutmix_alpha:    1.0 (default, 0 = kapalı)
      - mixup_prob:      Her batch'te mixup/cutmix uygulanma olasılığı
      - img_size:        224 (default) veya 384
      - branch_type:     'edge_texture' (default) veya 'color'
      - tta:             Test-Time Augmentation (predict_tta)
    """
    def __init__(self, device=None, model_type='efficientnet',
                 img_size=224, branch_type='edge_texture',
                 loss_type='focal', label_smoothing=0.1,
                 mixup_alpha=0.4, cutmix_alpha=1.0, mixup_prob=0.5,
                 center_guidance=False, center_guidance_strength=0.35,
                 center_guidance_sigma=None,
                 center_guidance_sigma_x=DEFAULT_CENTER_GUIDANCE_SIGMA_X,
                 center_guidance_sigma_y=DEFAULT_CENTER_GUIDANCE_SIGMA_Y,
                 center_guidance_center_x=DEFAULT_CENTER_GUIDANCE_CENTER_X,
                 center_guidance_center_y=DEFAULT_CENTER_GUIDANCE_CENTER_Y,
                 helmet_focus_loss_weight=DEFAULT_HELMET_FOCUS_LOSS_WEIGHT,
                 helmet_focus_sigma_x=DEFAULT_HELMET_FOCUS_SIGMA_X,
                 helmet_focus_sigma_y=DEFAULT_HELMET_FOCUS_SIGMA_Y,
                 helmet_focus_center_x=DEFAULT_HELMET_FOCUS_CENTER_X,
                 helmet_focus_center_y=DEFAULT_HELMET_FOCUS_CENTER_Y,
                 helmet_focus_outside_penalty=DEFAULT_HELMET_FOCUS_OUTSIDE_PENALTY):

        # ── Device ────────────────────────────────────────────────────────
        if device is None:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            except Exception as e:
                print(f"⚠️  CUDA error: {e}, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = device

        print(f"🖥️  Device: {self.device}")

        # ── Config ────────────────────────────────────────────────────────
        self.model_type = model_type
        self.img_size = img_size
        self.branch_type = branch_type
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.center_guidance = center_guidance
        self.center_guidance_strength = center_guidance_strength
        (
            self.center_guidance_sigma_x,
            self.center_guidance_sigma_y,
            self.center_guidance_center_x,
            self.center_guidance_center_y,
        ) = resolve_guidance_config(
            legacy_sigma=center_guidance_sigma,
            sigma_x=center_guidance_sigma_x,
            sigma_y=center_guidance_sigma_y,
            center_x=center_guidance_center_x,
            center_y=center_guidance_center_y,
        )
        self.center_guidance_sigma = legacy_guidance_sigma(
            self.center_guidance_sigma_x,
            self.center_guidance_sigma_y,
        )
        self.helmet_focus_loss_weight = float(helmet_focus_loss_weight)
        self.helmet_focus_sigma_x = float(helmet_focus_sigma_x)
        self.helmet_focus_sigma_y = float(helmet_focus_sigma_y)
        self.helmet_focus_center_x = float(helmet_focus_center_x)
        self.helmet_focus_center_y = float(helmet_focus_center_y)
        self.helmet_focus_outside_penalty = float(helmet_focus_outside_penalty)

        # ── Model ─────────────────────────────────────────────────────────
        self._build_model(pretrained=True)
        self.model.freeze_backbone()

        print(f"🧠 Model: HelmetClassifierNet v5 ({img_size}×{img_size})")
        print(f"   Branch: {branch_type} | Loss: {loss_type} | Smoothing: {label_smoothing}")
        print(f"   Mixup α={mixup_alpha} | CutMix α={cutmix_alpha} | prob={mixup_prob}")
        attention_mode = "ON" if center_guidance else "OFF"
        print(
            "   Guided attention: "
            f"{attention_mode} "
            f"(strength={center_guidance_strength}, sigma_x={self.center_guidance_sigma_x}, "
            f"sigma_y={self.center_guidance_sigma_y}, center=({self.center_guidance_center_x}, "
            f"{self.center_guidance_center_y}))"
        )
        print(
            "   Helmet focus loss: "
            f"weight={self.helmet_focus_loss_weight} "
            f"(sigma_x={self.helmet_focus_sigma_x}, sigma_y={self.helmet_focus_sigma_y}, "
            f"center=({self.helmet_focus_center_x}, {self.helmet_focus_center_y}))"
        )
        print("   Preprocess: aspect-ratio letterbox + tiny-safe augmentations")

        # ── Transforms ────────────────────────────────────────────────────
        self._configure_transforms()

        # ── History ───────────────────────────────────────────────────────
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _build_model(self, pretrained):
        self.model = HelmetClassifierNet(
            num_classes=2,
            pretrained=pretrained,
            branch_type=self.branch_type,
            center_guidance=self.center_guidance,
            center_guidance_strength=self.center_guidance_strength,
            center_guidance_sigma_x=self.center_guidance_sigma_x,
            center_guidance_sigma_y=self.center_guidance_sigma_y,
            center_guidance_center_x=self.center_guidance_center_x,
            center_guidance_center_y=self.center_guidance_center_y,
        ).to(self.device)

    def _compute_helmet_focus_loss(self, aux, labels):
        """Encourage helmet samples to activate on the upper helmet shell region."""
        if self.helmet_focus_loss_weight <= 0 or aux is None:
            return None

        helmet_mask = labels == 1
        if not torch.any(helmet_mask):
            return None

        losses = []
        for feature_map in aux['fpn_maps'].values():
            helmet_features = feature_map[helmet_mask]
            if helmet_features.numel() == 0:
                continue

            energy = helmet_features.abs().mean(dim=1, keepdim=True) + 1e-6
            distribution = energy / energy.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)

            prior = build_center_prior(
                distribution.shape[2],
                distribution.shape[3],
                distribution.device,
                distribution.dtype,
                sigma_x=self.helmet_focus_sigma_x,
                sigma_y=self.helmet_focus_sigma_y,
                center_x=self.helmet_focus_center_x,
                center_y=self.helmet_focus_center_y,
            )
            prior = prior / prior.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            prior = prior.expand(distribution.shape[0], -1, -1, -1)

            cross_entropy = -(prior * torch.log(distribution)).sum(dim=(2, 3))
            entropy_scale = max(float(np.log(max(distribution.shape[2] * distribution.shape[3], 2))), 1.0)
            cross_entropy = cross_entropy / entropy_scale
            outside_weight = (1.0 - prior / prior.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)).clamp_min(0.0)
            outside_energy = (distribution * outside_weight).sum(dim=(2, 3))
            losses.append((cross_entropy + self.helmet_focus_outside_penalty * outside_energy).mean())

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _configure_transforms(self):
        """
        Build transforms around aspect-ratio-preserving letterbox instead of
        stretch + random crop, which is harmful for tiny detections.
        """
        img_size = self.img_size

        self.train_transform = transforms.Compose([
            AspectRatioPadResize(img_size, scale_range=(0.92, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.06, 0.06),
                    scale=(0.95, 1.05),
                    fill=0,
                )
            ], p=0.6),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.15,
                    hue=0.05,
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.1),
            transforms.ToTensor(),
            self._normalize_transform(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])

        self.test_transform = transforms.Compose([
            AspectRatioPadResize(img_size),
            transforms.ToTensor(),
            self._normalize_transform(),
        ])

        self._tta_transforms = [
            self.test_transform,
            transforms.Compose([
                AspectRatioPadResize(img_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                self._normalize_transform(),
            ]),
            transforms.Compose([
                AspectRatioPadResize(img_size, scale=0.95),
                transforms.ToTensor(),
                self._normalize_transform(),
            ]),
            transforms.Compose([
                AspectRatioPadResize(img_size, scale=0.9),
                transforms.ToTensor(),
                self._normalize_transform(),
            ]),
        ]

    # ── Data ──────────────────────────────────────────────────────────────

    def prepare_data(self, dataset_dir, batch_size=32):
        """DataLoader'ları hazırlar."""
        print("📊 Dataset yükleniyor...")
        train_dataset = HelmetDataset(dataset_dir, split='train', transform=self.train_transform)
        val_dataset = HelmetDataset(dataset_dir, split='val', transform=self.test_transform)
        test_dataset = HelmetDataset(dataset_dir, split='test', transform=self.test_transform)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Val: {len(val_dataset)} samples")
        print(f"✓ Test: {len(test_dataset)} samples")

    # ── Loss ──────────────────────────────────────────────────────────────

    def _build_criterion(self):
        """Loss fonksiyonunu oluşturur."""
        if self.loss_type == 'focal':
            return FocalLoss(
                alpha=0.25, gamma=2.0,
                label_smoothing=self.label_smoothing
            )
        else:
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # ── Training ──────────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch_num, criterion, optimizer):
        """Tek epoch eğitim — Mixup/CutMix destekli."""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # ── Mixup / CutMix (olasılıkla) ──────────────────────────
            use_mix = np.random.rand() < self.mixup_prob
            if use_mix and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
                # %50 Mixup, %50 CutMix (ikisi de aktifse)
                if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
                    use_cutmix = np.random.rand() > 0.5
                elif self.cutmix_alpha > 0:
                    use_cutmix = True
                else:
                    use_cutmix = False

                if use_cutmix:
                    images, y_a, y_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                else:
                    images, y_a, y_b, lam = mixup_data(images, labels, self.mixup_alpha)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                loss.backward()
                optimizer.step()

                # Accuracy: karışık label'larla (en yüksek ağırlıklı label)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += (lam * predicted.eq(y_a).float()
                                  + (1 - lam) * predicted.eq(y_b).float()).sum().item()
            else:
                # Normal training step
                optimizer.zero_grad()
                need_aux = self.helmet_focus_loss_weight > 0 and torch.any(labels == 1)
                if need_aux:
                    outputs, aux = self.model(images, return_aux=True)
                else:
                    outputs = self.model(images)
                    aux = None
                loss = criterion(outputs, labels)
                focus_loss = self._compute_helmet_focus_loss(aux, labels)
                if focus_loss is not None:
                    loss = loss + self.helmet_focus_loss_weight * focus_loss
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })

        train_loss /= len(self.train_loader)
        train_acc = 100. * train_correct / train_total
        return train_loss, train_acc

    def validate(self):
        """Validation seti üzerinde değerlendirme."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        return val_loss / len(self.val_loader), 100. * val_correct / val_total

    def _save_checkpoint(self, checkpoint_path, epoch, phase, optimizer, scheduler,
                         best_val_acc, actual_phase1_epochs,
                         best_phase1_val_loss, no_improve_count,
                         num_epochs, learning_rate):
        """Eğitim durumunu checkpoint olarak kaydeder."""
        torch.save({
            # Model state
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # Training progress
            'epoch': epoch,
            'phase': phase,
            'best_val_acc': best_val_acc,
            'actual_phase1_epochs': actual_phase1_epochs,
            'best_phase1_val_loss': best_phase1_val_loss,
            'no_improve_count': no_improve_count,
            # History
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            # Config (resume sırasında doğrulama için)
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'model_type': self.model_type,
            'branch_type': self.branch_type,
            'img_size': self.img_size,
            'loss_type': self.loss_type,
            'label_smoothing': self.label_smoothing,
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
            'center_guidance': self.center_guidance,
            'center_guidance_strength': self.center_guidance_strength,
            'center_guidance_sigma': self.center_guidance_sigma,
            'center_guidance_sigma_x': self.center_guidance_sigma_x,
            'center_guidance_sigma_y': self.center_guidance_sigma_y,
            'center_guidance_center_x': self.center_guidance_center_x,
            'center_guidance_center_y': self.center_guidance_center_y,
            'helmet_focus_loss_weight': self.helmet_focus_loss_weight,
            'helmet_focus_sigma_x': self.helmet_focus_sigma_x,
            'helmet_focus_sigma_y': self.helmet_focus_sigma_y,
            'helmet_focus_center_x': self.helmet_focus_center_x,
            'helmet_focus_center_y': self.helmet_focus_center_y,
            'helmet_focus_outside_penalty': self.helmet_focus_outside_penalty,
        }, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """Checkpoint'tan eğitim durumunu yükler."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        saved_branch = ckpt.get('branch_type', self.branch_type)
        saved_img_size = ckpt.get('img_size', self.img_size)
        saved_center_guidance = ckpt.get('center_guidance', self.center_guidance)
        saved_center_guidance_strength = ckpt.get(
            'center_guidance_strength',
            self.center_guidance_strength,
        )
        has_explicit_guidance_shape = any(
            key in ckpt
            for key in (
                'center_guidance_sigma_x',
                'center_guidance_sigma_y',
                'center_guidance_center_x',
                'center_guidance_center_y',
            )
        )
        saved_center_guidance_sigma_x, saved_center_guidance_sigma_y, _, _ = resolve_guidance_config(
            legacy_sigma=ckpt.get('center_guidance_sigma'),
            sigma_x=ckpt.get('center_guidance_sigma_x'),
            sigma_y=ckpt.get('center_guidance_sigma_y'),
            center_x=0.0,
            center_y=0.0,
            default_sigma_x=self.center_guidance_sigma_x,
            default_sigma_y=self.center_guidance_sigma_y,
            default_center_x=0.0,
            default_center_y=0.0,
        )
        saved_center_guidance_center_x = ckpt.get('center_guidance_center_x', 0.0)
        saved_center_guidance_center_y = ckpt.get(
            'center_guidance_center_y',
            self.center_guidance_center_y if has_explicit_guidance_shape else 0.0,
        )
        saved_helmet_focus_loss_weight = ckpt.get('helmet_focus_loss_weight', 0.0)
        saved_helmet_focus_sigma_x = ckpt.get('helmet_focus_sigma_x', self.helmet_focus_sigma_x)
        saved_helmet_focus_sigma_y = ckpt.get('helmet_focus_sigma_y', self.helmet_focus_sigma_y)
        saved_helmet_focus_center_x = ckpt.get('helmet_focus_center_x', self.helmet_focus_center_x)
        saved_helmet_focus_center_y = ckpt.get('helmet_focus_center_y', self.helmet_focus_center_y)
        saved_helmet_focus_outside_penalty = ckpt.get(
            'helmet_focus_outside_penalty',
            self.helmet_focus_outside_penalty,
        )

        config_changed = (
            saved_branch != self.branch_type
            or saved_img_size != self.img_size
            or saved_center_guidance != self.center_guidance
            or saved_center_guidance_strength != self.center_guidance_strength
            or saved_center_guidance_sigma_x != self.center_guidance_sigma_x
            or saved_center_guidance_sigma_y != self.center_guidance_sigma_y
            or saved_center_guidance_center_x != self.center_guidance_center_x
            or saved_center_guidance_center_y != self.center_guidance_center_y
            or saved_helmet_focus_loss_weight != self.helmet_focus_loss_weight
            or saved_helmet_focus_sigma_x != self.helmet_focus_sigma_x
            or saved_helmet_focus_sigma_y != self.helmet_focus_sigma_y
            or saved_helmet_focus_center_x != self.helmet_focus_center_x
            or saved_helmet_focus_center_y != self.helmet_focus_center_y
            or saved_helmet_focus_outside_penalty != self.helmet_focus_outside_penalty
        )
        if config_changed:
            self.branch_type = saved_branch
            self.img_size = saved_img_size
            self.center_guidance = saved_center_guidance
            self.center_guidance_strength = saved_center_guidance_strength
            self.center_guidance_sigma_x = saved_center_guidance_sigma_x
            self.center_guidance_sigma_y = saved_center_guidance_sigma_y
            self.center_guidance_center_x = saved_center_guidance_center_x
            self.center_guidance_center_y = saved_center_guidance_center_y
            self.center_guidance_sigma = legacy_guidance_sigma(
                self.center_guidance_sigma_x,
                self.center_guidance_sigma_y,
            )
            self.helmet_focus_loss_weight = saved_helmet_focus_loss_weight
            self.helmet_focus_sigma_x = saved_helmet_focus_sigma_x
            self.helmet_focus_sigma_y = saved_helmet_focus_sigma_y
            self.helmet_focus_center_x = saved_helmet_focus_center_x
            self.helmet_focus_center_y = saved_helmet_focus_center_y
            self.helmet_focus_outside_penalty = saved_helmet_focus_outside_penalty
            self._build_model(pretrained=False)
            self._configure_transforms()

        # Model state yükle
        self.model.load_state_dict(ckpt['model_state_dict'])

        # History geri yükle
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.train_accs = ckpt.get('train_accs', [])
        self.val_accs = ckpt.get('val_accs', [])

        return ckpt

    def train(self, num_epochs=30, learning_rate=0.001,
              output_dir='.', model_name='helmet_classifier_v5.pth',
              resume=False):
        """
        2-fazlı transfer learning ile eğitim.
          Phase 1: Backbone frozen, head eğitimi (patience-based early stop)
          Phase 2: Full fine-tuning (düşük lr)

        Args:
            resume: True ise output_dir/checkpoint_last.pth'den devam eder
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = str(output_dir / model_name)
        checkpoint_path = str(output_dir / 'checkpoint_last.pth')

        print(f"\n📂 Output dizini: {output_dir.resolve()}")

        criterion = self._build_criterion()
        best_val_acc = 0.0
        total = sum(p.numel() for p in self.model.parameters())

        patience = 3
        min_phase1 = 3
        max_phase1 = num_epochs // 2

        # ── Resume kontrolü ──────────────────────────────────────────
        resume_epoch = 0
        resume_phase = 1
        actual_phase1_epochs = max_phase1
        best_phase1_val_loss = float('inf')
        no_improve_count = 0
        phase1_early_stopped = False

        if resume and Path(checkpoint_path).exists():
            print(f"\n🔄 Checkpoint bulundu: {checkpoint_path}")
            ckpt = self._load_checkpoint(checkpoint_path)

            resume_epoch = ckpt['epoch']
            resume_phase = ckpt['phase']
            best_val_acc = ckpt['best_val_acc']
            actual_phase1_epochs = ckpt.get('actual_phase1_epochs', max_phase1)
            best_phase1_val_loss = ckpt.get('best_phase1_val_loss', float('inf'))
            no_improve_count = ckpt.get('no_improve_count', 0)

            print(f"   Epoch {resume_epoch}/{num_epochs} | Phase {resume_phase}")
            print(f"   Best Val Acc: {best_val_acc:.2f}%")
            print(f"   History: {len(self.train_losses)} epochs kaydedilmiş")

            # Phase 1 bitmişse (resume_phase == 2) → doğrudan phase 2'ye atla
            if resume_phase == 2:
                phase1_early_stopped = True
                actual_phase1_epochs = ckpt['actual_phase1_epochs']
        elif resume:
            print(f"\n⚠️  Checkpoint bulunamadı: {checkpoint_path}")
            print("   Sıfırdan başlanıyor...")

        # ── Phase 1: Backbone frozen ─────────────────────────────────
        if not phase1_early_stopped and resume_epoch < max_phase1:
            self.model.freeze_backbone()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            phase1_start = resume_epoch if (resume and resume_phase == 1) else 0

            print(f"\n🎯 Phase 1: Backbone frozen (max {max_phase1} ep, patience={patience})")
            if phase1_start > 0:
                print(f"   ▶ Epoch {phase1_start + 1}'den devam ediliyor")
            print(f"   Trainable: {trainable:,} / {total:,}")
            print(f"   Loss: {self.loss_type} | Smoothing: {self.label_smoothing}")
            print("=" * 60)

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate, weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_phase1)

            # Resume: optimizer ve scheduler state'ini geri yükle
            if resume and resume_phase == 1 and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            for epoch in range(phase1_start, max_phase1):
                global_epoch = epoch + 1
                train_loss, train_acc = self._train_one_epoch(global_epoch, criterion, optimizer)
                val_loss, val_acc = self.validate()

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                scheduler.step()

                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {global_epoch}")
                print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, LR: {lr:.6f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save(best_model_path)
                    print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")

                if val_loss < best_phase1_val_loss:
                    best_phase1_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Checkpoint kaydet
                self._save_checkpoint(
                    checkpoint_path, global_epoch, 1, optimizer, scheduler,
                    best_val_acc, actual_phase1_epochs,
                    best_phase1_val_loss, no_improve_count,
                    num_epochs, learning_rate,
                )
                print(f"  💾 Checkpoint saved")

                if epoch >= min_phase1 and no_improve_count >= patience:
                    actual_phase1_epochs = epoch + 1
                    print(f"\n⏹️  Phase 1 early stopped at epoch {actual_phase1_epochs}")
                    break

                print("-" * 60)
            else:
                actual_phase1_epochs = max_phase1

        # ── Phase 2: Full fine-tuning ────────────────────────────────
        phase2_total = num_epochs - actual_phase1_epochs
        phase2_lr = learning_rate * 0.1

        # Resume: phase 2'den devam edeceksek başlangıç epoch hesapla
        if resume and resume_phase == 2:
            phase2_start = resume_epoch - actual_phase1_epochs
        else:
            phase2_start = 0

        print(f"\n🎯 Phase 2: Full fine-tuning ({phase2_total} ep, lr={phase2_lr})")
        if phase2_start > 0:
            print(f"   ▶ Phase 2 epoch {phase2_start + 1}'den devam ediliyor")
        self.model.unfreeze_backbone()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Trainable: {trainable:,} / {total:,}")
        print("=" * 60)

        optimizer = optim.Adam(self.model.parameters(), lr=phase2_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_total)

        # Resume: optimizer ve scheduler state'ini geri yükle
        if resume and resume_phase == 2 and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        for epoch in range(phase2_start, phase2_total):
            global_epoch = actual_phase1_epochs + epoch + 1
            train_loss, train_acc = self._train_one_epoch(global_epoch, criterion, optimizer)
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {global_epoch}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, LR: {lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(best_model_path)
                print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")

            # Checkpoint kaydet
            self._save_checkpoint(
                checkpoint_path, global_epoch, 2, optimizer, scheduler,
                best_val_acc, actual_phase1_epochs,
                best_phase1_val_loss, no_improve_count,
                num_epochs, learning_rate,
            )
            print(f"  💾 Checkpoint saved")
            print("-" * 60)

        print(f"\n✅ Training tamamlandı! Best Val Acc: {best_val_acc:.2f}%")
        # Training history'yi model ile aynı klasöre kaydet
        model_dir = Path(best_model_path).parent
        self.plot_training_history(model_dir)
        return best_model_path

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, model_path=None):
        """Test seti üzerinde değerlendirme."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        print("\n📊 Test seti değerlendiriliyor...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        print(f"\n{'='*60}")
        print(f"📊 TEST SONUÇLARI")
        print(f"{'='*60}")
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=['no_helmet', 'helmet']))
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        # Confusion matrix'i model ile aynı klasöre kaydet
        if model_path:
            output_dir = Path(model_path).parent
        else:
            output_dir = Path('.')
        self.plot_confusion_matrix(cm, output_dir)

        return {
            'accuracy': acc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'confusion_matrix': cm
        }

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, image_path, return_proba=False):
        """Tek görüntü tahmini."""
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        tensor = self.test_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()

        if return_proba:
            return {
                'prediction': pred,
                'confidence': float(probs[pred]),
                'probabilities': {
                    'no_helmet': float(probs[0]),
                    'helmet': float(probs[1])
                }
            }
        return pred

    def predict_tta(self, image_input, return_proba=False):
        """
        Test-Time Augmentation ile tahmin.

        4 farklı görünüm üzerinde tahmin yapıp ortalamasını alır.
        Overconfidence azaltır, borderline vakalarda doğruluğu artırır.

        Args:
            image_input: PIL Image veya dosya yolu
            return_proba: True ise detaylı sonuç döner
        """
        self.model.eval()

        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input

        all_probs = []
        for t in self._tta_transforms:
            tensor = t(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)

        # Ortalamalı olasılık
        avg_probs = torch.stack(all_probs).mean(dim=0)[0]
        pred = avg_probs.argmax().item()

        if return_proba:
            return {
                'prediction': pred,
                'confidence': float(avg_probs[pred]),
                'probabilities': {
                    'no_helmet': float(avg_probs[0]),
                    'helmet': float(avg_probs[1])
                },
                'tta_views': len(self._tta_transforms)
            }
        return pred

    def predict_batch(self, images, return_proba=False):
        """Batch tahmin — pipeline uyumlu."""
        self.model.eval()
        batch_tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            batch_tensors.append(self.test_transform(img))

        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

        results = []
        for i in range(len(images)):
            if return_proba:
                results.append({
                    'prediction': int(preds[i]),
                    'confidence': float(probs[i][preds[i]]),
                    'probabilities': {
                        'no_helmet': float(probs[i][0]),
                        'helmet': float(probs[i][1])
                    }
                })
            else:
                results.append(int(preds[i]))
        return results

    def predict_batch_tta(self, images, return_proba=False):
        """Batch TTA tahmini — her görüntü için çoklu görünüm."""
        results = []
        for img in images:
            r = self.predict_tta(img, return_proba=return_proba)
            results.append(r)
        return results

    # ── Grad-CAM ──────────────────────────────────────────────────────────

    def gradcam(self, image_input, target_class=None, target_layer=None):
        """
        Grad-CAM heatmap üretir — modelin görüntünün neresine baktığını gösterir.

        Args:
            image_input: PIL Image veya dosya yolu
            target_class: Hedef sınıf (None = predicted class)
            target_layer: Hook katmanı (None = fpn_cbam3)

        Returns:
            dict: {
                'heatmap':     np.array (H, W) 0-1 float — raw heatmap
                'overlay':     np.array (H, W, 3) uint8  — görüntü üzerine overlay
                'prediction':  int
                'confidence':  float
                'probabilities': dict
            }
        """
        import cv2

        self.model.eval()

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert('RGB')
        else:
            pil_img = image_input

        # Hook kaydet
        hooks = self.model.register_gradcam_hooks(target_layer)

        # Forward
        tensor = self.test_transform(pil_img).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = output.argmax(1).item()

        if target_class is None:
            target_class = pred

        # Backward (hedef sınıf için gradient)
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM hesapla
        gradients = self.model._gradcam_gradients  # (1, C, H, W)
        activations = self.model._gradcam_activations  # (1, C, H, W)

        weights = gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * activations).sum(dim=1, keepdim=True)  # Weighted sum
        cam = F.relu(cam)  # ReLU
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        img_np = np.array(pil_img)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        # Overlay oluştur
        heatmap_colored = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * img_np + 0.4 * heatmap_colored).astype(np.uint8)

        # Hook temizle
        for h in hooks:
            h.remove()
        self.model._gradcam_activations = None
        self.model._gradcam_gradients = None

        return {
            'heatmap': cam_resized,
            'overlay': overlay,
            'prediction': pred,
            'confidence': float(probs[pred]),
            'probabilities': {
                'no_helmet': float(probs[0]),
                'helmet': float(probs[1])
            }
        }

    def gradcam_comparison(self, image_input, save_path=None):
        """
        Birden fazla katmanda Grad-CAM karşılaştırması yapar.
        FPN katmanları + side branch'i karşılaştırır.

        Args:
            image_input: PIL Image veya dosya yolu
            save_path: Kaydedilecek dosya yolu (None = göster)

        Returns:
            fig: matplotlib Figure
        """
        import cv2

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert('RGB')
        else:
            pil_img = image_input

        layers = {
            'Stage 3 (high-level)': self.model.cbam3,
            'FPN P5 (coarse)': self.model.fpn_cbam3,
            'FPN P4 (mid)': self.model.fpn_cbam2,
            'FPN P3 (fine)': self.model.fpn_cbam1,
        }

        fig, axes = plt.subplots(1, len(layers) + 1, figsize=(4 * (len(layers) + 1), 4))

        # Original
        axes[0].imshow(pil_img)
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i, (name, layer) in enumerate(layers.items()):
            result = self.gradcam(pil_img, target_layer=layer)
            axes[i + 1].imshow(result['overlay'])
            axes[i + 1].set_title(f'{name}\n{result["confidence"]:.1%}')
            axes[i + 1].axis('off')

        pred_label = 'helmet' if result['prediction'] == 1 else 'no_helmet'
        fig.suptitle(f'Grad-CAM: {pred_label} ({result["confidence"]:.1%})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Grad-CAM kaydedildi: {save_path}")
        plt.close()
        return fig

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, filepath):
        """Modeli kaydeder."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'branch_type': self.branch_type,
            'img_size': self.img_size,
            'loss_type': self.loss_type,
            'label_smoothing': self.label_smoothing,
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
            'center_guidance': self.center_guidance,
            'center_guidance_strength': self.center_guidance_strength,
            'center_guidance_sigma': self.center_guidance_sigma,
            'center_guidance_sigma_x': self.center_guidance_sigma_x,
            'center_guidance_sigma_y': self.center_guidance_sigma_y,
            'center_guidance_center_x': self.center_guidance_center_x,
            'center_guidance_center_y': self.center_guidance_center_y,
            'helmet_focus_loss_weight': self.helmet_focus_loss_weight,
            'helmet_focus_sigma_x': self.helmet_focus_sigma_x,
            'helmet_focus_sigma_y': self.helmet_focus_sigma_y,
            'helmet_focus_center_x': self.helmet_focus_center_x,
            'helmet_focus_center_y': self.helmet_focus_center_y,
            'helmet_focus_outside_penalty': self.helmet_focus_outside_penalty,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, filepath)

    def load(self, filepath):
        """
        Modeli yükler. v4 (color branch) ve v5 (edge_texture) uyumlu.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        saved_type = checkpoint.get('model_type', 'efficientnet')
        saved_branch = checkpoint.get('branch_type', 'color')  # v4 default
        saved_img_size = checkpoint.get('img_size', 224)
        saved_center_guidance = checkpoint.get('center_guidance', False)
        saved_center_guidance_strength = checkpoint.get('center_guidance_strength', 0.35)
        has_explicit_guidance_shape = any(
            key in checkpoint
            for key in (
                'center_guidance_sigma_x',
                'center_guidance_sigma_y',
                'center_guidance_center_x',
                'center_guidance_center_y',
            )
        )
        saved_center_guidance_sigma_x, saved_center_guidance_sigma_y, _, _ = resolve_guidance_config(
            legacy_sigma=checkpoint.get('center_guidance_sigma'),
            sigma_x=checkpoint.get('center_guidance_sigma_x'),
            sigma_y=checkpoint.get('center_guidance_sigma_y'),
            center_x=0.0,
            center_y=0.0,
            default_sigma_x=DEFAULT_CENTER_GUIDANCE_SIGMA_X,
            default_sigma_y=DEFAULT_CENTER_GUIDANCE_SIGMA_Y,
            default_center_x=0.0,
            default_center_y=0.0,
        )
        saved_center_guidance_center_x = checkpoint.get('center_guidance_center_x', 0.0)
        saved_center_guidance_center_y = checkpoint.get(
            'center_guidance_center_y',
            DEFAULT_CENTER_GUIDANCE_CENTER_Y if has_explicit_guidance_shape else 0.0,
        )
        saved_helmet_focus_loss_weight = checkpoint.get('helmet_focus_loss_weight', 0.0)
        saved_helmet_focus_sigma_x = checkpoint.get('helmet_focus_sigma_x', DEFAULT_HELMET_FOCUS_SIGMA_X)
        saved_helmet_focus_sigma_y = checkpoint.get('helmet_focus_sigma_y', DEFAULT_HELMET_FOCUS_SIGMA_Y)
        saved_helmet_focus_center_x = checkpoint.get('helmet_focus_center_x', DEFAULT_HELMET_FOCUS_CENTER_X)
        saved_helmet_focus_center_y = checkpoint.get('helmet_focus_center_y', DEFAULT_HELMET_FOCUS_CENTER_Y)
        saved_helmet_focus_outside_penalty = checkpoint.get(
            'helmet_focus_outside_penalty',
            DEFAULT_HELMET_FOCUS_OUTSIDE_PENALTY,
        )

        if saved_type != 'efficientnet':
            raise ValueError(f"Unsupported model_type='{saved_type}'.")

        # Branch type ve img_size güncelle
        self.branch_type = saved_branch
        self.img_size = saved_img_size
        self.center_guidance = saved_center_guidance
        self.center_guidance_strength = saved_center_guidance_strength
        self.center_guidance_sigma_x = saved_center_guidance_sigma_x
        self.center_guidance_sigma_y = saved_center_guidance_sigma_y
        self.center_guidance_center_x = saved_center_guidance_center_x
        self.center_guidance_center_y = saved_center_guidance_center_y
        self.center_guidance_sigma = legacy_guidance_sigma(
            self.center_guidance_sigma_x,
            self.center_guidance_sigma_y,
        )
        self.helmet_focus_loss_weight = saved_helmet_focus_loss_weight
        self.helmet_focus_sigma_x = saved_helmet_focus_sigma_x
        self.helmet_focus_sigma_y = saved_helmet_focus_sigma_y
        self.helmet_focus_center_x = saved_helmet_focus_center_x
        self.helmet_focus_center_y = saved_helmet_focus_center_y
        self.helmet_focus_outside_penalty = saved_helmet_focus_outside_penalty

        self._build_model(pretrained=False)

        self._configure_transforms()

        # v4 → v5 state_dict uyumu: color_branch → side_branch
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # v4 checkpoint'ta color_branch varsa side_branch'e map et
            new_key = k.replace('color_branch.', 'side_branch.')
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])

        print(f"✓ Model yüklendi: {filepath}")
        print(f"  Branch: {saved_branch} | Size: {saved_img_size} | "
              f"Epochs: {len(self.train_losses)}")
        print(
            "  Guided attention: "
            f"{'ON' if self.center_guidance else 'OFF'} "
            f"(strength={self.center_guidance_strength}, sigma_x={self.center_guidance_sigma_x}, "
            f"sigma_y={self.center_guidance_sigma_y}, center=({self.center_guidance_center_x}, "
            f"{self.center_guidance_center_y}))"
        )
        print(
            "  Helmet focus loss: "
            f"weight={self.helmet_focus_loss_weight} "
            f"(sigma_x={self.helmet_focus_sigma_x}, sigma_y={self.helmet_focus_sigma_y}, "
            f"center=({self.helmet_focus_center_x}, {self.helmet_focus_center_y}))"
        )

    # ── Visualization ─────────────────────────────────────────────────────

    def plot_training_history(self, output_dir='.'):
        """Training history görselleştirir."""
        output_dir = Path(output_dir)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = output_dir / 'training_history.png'
        plt.savefig(str(save_path), dpi=150)
        print(f"✓ Training history: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm, output_dir='.'):
        """Confusion matrix görselleştirir."""
        output_dir = Path(output_dir)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['no_helmet', 'helmet'],
                    yticklabels=['no_helmet', 'helmet'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = output_dir / 'confusion_matrix.png'
        plt.savefig(str(save_path), dpi=150)
        print(f"✓ Confusion matrix: {save_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Helmet Classifier v5 Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224, choices=[224, 384])
    parser.add_argument('--branch', type=str, default='edge_texture',
                        choices=['edge_texture', 'color'])
    parser.add_argument('--disable-center-guidance', action='store_true')
    parser.add_argument('--center-guidance-strength', type=float, default=0.35)
    parser.add_argument('--center-guidance-sigma', type=float, default=None,
                        help='Legacy isotropic sigma. Verilirse sigma_x/sigma_y yerine kullanılır.')
    parser.add_argument('--center-guidance-sigma-x', type=float, default=DEFAULT_CENTER_GUIDANCE_SIGMA_X)
    parser.add_argument('--center-guidance-sigma-y', type=float, default=DEFAULT_CENTER_GUIDANCE_SIGMA_Y)
    parser.add_argument('--center-guidance-center-x', type=float, default=DEFAULT_CENTER_GUIDANCE_CENTER_X)
    parser.add_argument('--center-guidance-center-y', type=float, default=DEFAULT_CENTER_GUIDANCE_CENTER_Y)
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce'])
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.4)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--helmet-focus-loss-weight', type=float, default=DEFAULT_HELMET_FOCUS_LOSS_WEIGHT)
    parser.add_argument('--helmet-focus-sigma-x', type=float, default=DEFAULT_HELMET_FOCUS_SIGMA_X)
    parser.add_argument('--helmet-focus-sigma-y', type=float, default=DEFAULT_HELMET_FOCUS_SIGMA_Y)
    parser.add_argument('--helmet-focus-center-x', type=float, default=DEFAULT_HELMET_FOCUS_CENTER_X)
    parser.add_argument('--helmet-focus-center-y', type=float, default=DEFAULT_HELMET_FOCUS_CENTER_Y)
    parser.add_argument('--helmet-focus-outside-penalty', type=float, default=DEFAULT_HELMET_FOCUS_OUTSIDE_PENALTY)
    parser.add_argument('--output-dir', type=str, default='models/trained')
    parser.add_argument('--model-name', type=str, default='helmet_classifier_v5.pth')
    args = parser.parse_args()

    print("=" * 70)
    print(" " * 15 + "HELMET CLASSIFIER v5 TRAINING")
    print("=" * 70)

    classifier = CNNClassifier(
        img_size=args.img_size,
        branch_type=args.branch,
        loss_type=args.loss,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        center_guidance=not args.disable_center_guidance,
        center_guidance_strength=args.center_guidance_strength,
        center_guidance_sigma=args.center_guidance_sigma,
        center_guidance_sigma_x=args.center_guidance_sigma_x,
        center_guidance_sigma_y=args.center_guidance_sigma_y,
        center_guidance_center_x=args.center_guidance_center_x,
        center_guidance_center_y=args.center_guidance_center_y,
        helmet_focus_loss_weight=args.helmet_focus_loss_weight,
        helmet_focus_sigma_x=args.helmet_focus_sigma_x,
        helmet_focus_sigma_y=args.helmet_focus_sigma_y,
        helmet_focus_center_x=args.helmet_focus_center_x,
        helmet_focus_center_y=args.helmet_focus_center_y,
        helmet_focus_outside_penalty=args.helmet_focus_outside_penalty,
    )

    total_params = sum(p.numel() for p in classifier.model.parameters())
    trainable = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"📐 Total: {total_params:,} | Trainable: {trainable:,}")
    print("=" * 70)

    classifier.prepare_data(args.dataset, batch_size=args.batch_size)

    best_path = classifier.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    classifier.load(best_path)
    results = classifier.evaluate(model_path=best_path)

    print(f"\n{'='*70}")
    print(f"✅ TRAINING TAMAMLANDI — Test Acc: {results['accuracy']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
