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

class HelmetDataset(Dataset):
    """
    Helmet/No-Helmet dataset için PyTorch Dataset.
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Dataset ana klasörü
            split: 'train', 'val', veya 'test'
            transform: Torchvision transforms
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []
        
        helmet_dir = self.data_dir / 'helmet'
        no_helmet_dir = self.data_dir / 'no_helmet'
        
        for img_path in helmet_dir.glob('*.png'):
            self.samples.append(img_path)
            self.labels.append(1)
        for img_path in helmet_dir.glob('*.jpg'):
            self.samples.append(img_path)
            self.labels.append(1)
        
        for img_path in no_helmet_dir.glob('*.png'):
            self.samples.append(img_path)
            self.labels.append(0)
        for img_path in no_helmet_dir.glob('*.jpg'):
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


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 — öğrenilebilir offset ile geometrik dönüşümlere
    uyum sağlar. Smooth katmanları yerine kullanılır.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Offset: 2 * kernel_size^2 (x,y per kernel position)
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
        # Modulation mask: kernel_size^2
        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.5)
        
        # Actual conv weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight)
        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
    
    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(
            x, offset, self.weight,
            bias=self.bias,
            padding=self.padding,
            mask=mask
        )


class AdaptiveFeaturePooling(nn.Module):
    """
    Adaptive Feature Pooling — her FPN ölçeğine öğrenilebilir ağırlık verir.
    
    Model her görüntü için hangi ölçeğin (p3, p4, p5) daha önemli olduğunu
    dinamik olarak öğrenir. Attention-based weighted fusion.
    """
    def __init__(self, feature_dim=128, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # Attention network: tüm ölçekleri birleştirip her birine ağırlık hesaplar
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_scales),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of [p3_pool, p4_pool, p5_pool] — her biri (B, 128)
        Returns:
            weighted_feat: (B, 128) — ağırlıklı toplam
        """
        # Stack all features: (B, num_scales, feature_dim)
        stacked = torch.stack(features, dim=1)  # (B, 3, 128)
        
        # Concat for attention: (B, num_scales * feature_dim)
        concat = stacked.view(stacked.size(0), -1)  # (B, 384)
        
        # Compute attention weights: (B, num_scales)
        weights = self.attention(concat)  # (B, 3)
        
        # Weighted sum: (B, feature_dim)
        weights = weights.unsqueeze(2)  # (B, 3, 1)
        weighted_feat = (stacked * weights).sum(dim=1)  # (B, 128)
        
        return weighted_feat


class HelmetClassifierNet(nn.Module):
    """
    Production-grade helmet classifier.

    EfficientNet-B0 backbone (pretrained ImageNet) + CBAM + FPN + Color Branch.

    Mimari:
      - Backbone: EfficientNet-B0 features, 3 ölçek çıkarma (40ch, 112ch, 1280ch)
      - Attention: Her ölçekte CBAM (Channel + Spatial)
      - FPN: Top-down lateral bağlantılar, tüm ölçekler 128-dim'e projekte
      - Color Branch: Öğrenilen renk uzayı dönüşümü (1x1 conv) → 64-dim
      - Classifier: (128 + 64) = 192 → 128 → 64 → 2

    Giriş: 224x224 RGB
    ~5.9M parametre (5.3M backbone, ~600K trainable head)
    2-fazlı eğitim: Phase 1 backbone frozen, Phase 2 full fine-tune
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.stage1 = backbone.features[:4]   # → 40ch,  28x28  (texture)
        self.stage2 = backbone.features[4:6]  # → 112ch, 14x14  (mid semantics)
        self.stage3 = backbone.features[6:]   # → 1280ch, 7x7   (high semantics)

        self.cbam1 = CBAM(40, reduction=8)
        self.cbam2 = CBAM(112, reduction=16)
        self.cbam3 = CBAM(1280, reduction=16)

        self.lateral3 = nn.Conv2d(1280, 128, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(112, 128, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(40, 128, kernel_size=1, bias=False)
        
        # Deformable Conv replaces standard smooth layers
        self.smooth2 = DeformableConv2d(128, 128, kernel_size=3, padding=1)
        self.smooth1 = DeformableConv2d(128, 128, kernel_size=3, padding=1)
        
        # Post-FPN CBAM — refine FPN outputs before pooling
        self.fpn_cbam3 = CBAM(128, reduction=8)
        self.fpn_cbam2 = CBAM(128, reduction=8)
        self.fpn_cbam1 = CBAM(128, reduction=8)

        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                                    # 224→112 early downsampling
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                                    # 112→56 (was AvgPool2d(4))
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                            # 56→1
        )
        
        # Adaptive Feature Pooling for FPN fusion
        self.adaptive_pooling = AdaptiveFeaturePooling(feature_dim=128, num_scales=3)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),      # was 0.4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),      # was 0.3
            nn.Linear(64, num_classes)
        )

    def freeze_backbone(self):
        """Backbone parametrelerini dondurur (transfer learning Phase 1)."""
        for stage in [self.stage1, self.stage2, self.stage3]:
            for param in stage.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Backbone parametrelerini açar (transfer learning Phase 2)."""
        for stage in [self.stage1, self.stage2, self.stage3]:
            for param in stage.parameters():
                param.requires_grad = True

    def forward(self, x):
        color_feat = self.color_branch(x).view(x.size(0), -1)

        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        c3 = self.cbam1(c3)
        c4 = self.cbam2(c4)
        c5 = self.cbam3(c5)

        p5 = self.lateral3(c5)
        p4 = self.smooth2(self.lateral2(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest'))
        p3 = self.smooth1(self.lateral1(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest'))

        # Post-FPN CBAM (second attention round)
        p5 = self.fpn_cbam3(p5)
        p4 = self.fpn_cbam2(p4)
        p3 = self.fpn_cbam1(p3)

        p5_pool = F.adaptive_avg_pool2d(p5, 1).view(x.size(0), -1)
        p4_pool = F.adaptive_avg_pool2d(p4, 1).view(x.size(0), -1)
        p3_pool = F.adaptive_avg_pool2d(p3, 1).view(x.size(0), -1)
        
        # Adaptive weighted fusion (replaces simple addition)
        fpn_feat = self.adaptive_pooling([p3_pool, p4_pool, p5_pool])

        fused = torch.cat([fpn_feat, color_feat], dim=1)
        return self.classifier(fused)


class CNNClassifier:
    """
    CNN-based helmet classifier.
    """
    def __init__(self, device=None, model_type='efficientnet'):
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
        
        self.model_type = model_type
        
        self.model = HelmetClassifierNet(num_classes=2, pretrained=True).to(self.device)
        self.model.freeze_backbone()
        img_size = 224
        print(f"🧠 Model: HelmetClassifierNet (EfficientNet-B0 + CBAM + FPN + Color, {img_size}x{img_size})")
        
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def prepare_data(self, dataset_dir, batch_size=32):
        """
        DataLoader'ları hazırlar.
        """
        print("📊 Dataset yükleniyor...")
        
        train_dataset = HelmetDataset(dataset_dir, split='train', transform=self.train_transform)
        val_dataset = HelmetDataset(dataset_dir, split='val', transform=self.test_transform)
        test_dataset = HelmetDataset(dataset_dir, split='test', transform=self.test_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Val: {len(val_dataset)} samples")
        print(f"✓ Test: {len(test_dataset)} samples")
    
    def _run_epochs(self, num_epochs, criterion, optimizer, scheduler, best_val_acc,
                    best_model_path, epoch_offset=0):
        """
        Epoch döngüsü — train() tarafından çağrılır.
        """
        for epoch in range(num_epochs):
            global_epoch = epoch_offset + epoch + 1
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {global_epoch}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {global_epoch}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(best_model_path)
                print(f"  ✓ Best model saved: {best_model_path} (Val Acc: {val_acc:.2f}%)")
            
            print("-"*60)
        
        return best_val_acc

    def train(self, num_epochs=20, learning_rate=0.001,
              output_dir='.', model_name='cnn_classifier_best.pth'):
        """
        Modeli eğitir.
        EfficientNet modeli için 2-fazlı transfer learning uygular:
          Phase 1: Backbone frozen, sadece head eğitilir (patience-based early stop)
          Phase 2: Tüm ağ fine-tune edilir (düşük lr, kalan epoch'lar)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = str(output_dir / model_name)

        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        
        if self.model_type == 'efficientnet':
            # Phase 1: patience-based early stopping
            patience = 3
            min_phase1 = 3
            max_phase1 = num_epochs // 2
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            
            print(f"\n🎯 Phase 1: Backbone frozen (max {max_phase1} epoch, patience={patience}, lr={learning_rate})")
            print(f"   Eğitilebilir: {trainable:,} / {total:,} parametre")
            print("="*60)
            
            self.model.freeze_backbone()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate, weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_phase1)
            
            best_phase1_val_loss = float('inf')
            no_improve_count = 0
            actual_phase1_epochs = max_phase1
            
            for epoch in range(max_phase1):
                global_epoch = epoch + 1
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                pbar = tqdm(self.train_loader, desc=f'Epoch {global_epoch}')
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
                
                train_loss = train_loss / len(self.train_loader)
                train_acc = 100. * train_correct / train_total
                val_loss, val_acc = self.validate()
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {global_epoch}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save(best_model_path)
                    print(f"  ✓ Best model saved: {best_model_path} (Val Acc: {val_acc:.2f}%)")
                
                # Early stopping check
                if val_loss < best_phase1_val_loss:
                    best_phase1_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if epoch >= min_phase1 and no_improve_count >= patience:
                    actual_phase1_epochs = epoch + 1
                    print(f"\n⏹️  Phase 1 early stopped at epoch {actual_phase1_epochs} "
                          f"(no val_loss improvement for {patience} epochs)")
                    break
                
                print("-"*60)
            else:
                actual_phase1_epochs = max_phase1
            
            # Phase 2: full fine-tuning with remaining epochs
            phase2_epochs = num_epochs - actual_phase1_epochs
            
            print(f"\n🎯 Phase 2: Full fine-tuning ({phase2_epochs} epoch, lr={learning_rate * 0.1})")
            self.model.unfreeze_backbone()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"   Eğitilebilir: {trainable:,} / {total:,} parametre")
            print("="*60)
            
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
            best_val_acc = self._run_epochs(
                phase2_epochs, criterion, optimizer, scheduler,
                best_val_acc, best_model_path,
                epoch_offset=actual_phase1_epochs
            )
        else:
            print(f"\n🎯 CNN modeli eğitiliyor...")
            print(f"Epochs: {num_epochs}, Learning Rate: {learning_rate}")
            print("="*60)
            
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
            best_val_acc = self._run_epochs(
                num_epochs, criterion, optimizer, scheduler,
                best_val_acc, best_model_path
            )
        
        print(f"\n✅ Training tamamlandı! Best Val Acc: {best_val_acc:.2f}%")
        print(f"✓ En iyi model yolu: {best_model_path}")
        self.plot_training_history()
        return best_model_path
    
    def validate(self):
        """
        Validation seti üzerinde değerlendirme yapar.
        """
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
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def evaluate(self):
        """
        Test seti üzerinde değerlendirme yapar.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
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
        
        print("\n" + "="*60)
        print("📊 TEST SONUÇLARI")
        print("="*60)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\nAccuracy: {acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['no_helmet', 'helmet']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        self.plot_confusion_matrix(cm)
        
        return {
            'accuracy': acc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """
        Training history görselleştirir.
        """
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
        plt.savefig('cnn_classifier_training_history.png', dpi=150)
        print(f"\n✓ Training history kaydedildi: cnn_classifier_training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """
        Confusion matrix görselleştirir.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['no_helmet', 'helmet'],
                    yticklabels=['no_helmet', 'helmet'])
        plt.title('CNN Classifier - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('cnn_classifier_confusion_matrix.png', dpi=150)
        print(f"✓ Confusion matrix kaydedildi: cnn_classifier_confusion_matrix.png")
        plt.close()
    
    def predict(self, image_path, return_proba=False):
        """
        Tek bir görüntü için tahmin yapar.
        """
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()
        
        if return_proba:
            return {
                'prediction': pred,
                'confidence': float(probs[pred]),
                'probabilities': {'no_helmet': float(probs[0]), 'helmet': float(probs[1])}
            }
        else:
            return pred
    
    def predict_batch(self, images, return_proba=False):
        """
        Birden fazla görüntü için batch tahmin yapar.
        
        Args:
            images: List of PIL Images veya list of image paths
            return_proba: True ise olasılık dön
        Returns:
            List of predictions
        """
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
    
    def save(self, filepath):
        """
        Modeli kaydeder.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, filepath)
    
    def load(self, filepath):
        """
        Modeli yükler.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        saved_type = checkpoint.get('model_type', 'efficientnet')
        if saved_type != self.model_type:
            print(f"⚠️  Checkpoint model_type='{saved_type}', yeniden oluşturuluyor...")
            self.model_type = saved_type
        
        if saved_type != 'efficientnet':
            raise ValueError(
                f"Unsupported model_type='{saved_type}'. "
                "Only 'efficientnet' (HelmetClassifierNet) is supported. "
                "Legacy models (lightweight, color_aware) are deprecated."
            )
        
        self.model = HelmetClassifierNet(num_classes=2, pretrained=False).to(self.device)
        img_size = 224
        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        print(f"✓ Model yüklendi: {filepath}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CNN Helmet Classifier Training')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet'],
                        help='Model mimarisi: efficientnet (HelmetClassifierNet: EfficientNet-B0 + CBAM + FPN + Color)')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch sayısı')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', type=str,
                        default='/home/berhan/Development/personal/HelmetClassCorrector/dataset',
                        help='Dataset klasörü')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Eğitim çıktılarının kaydedileceği klasör')
    parser.add_argument('--model-name', type=str, default='cnn_classifier_best.pth',
                        help='Kaydedilecek en iyi model dosya adı')
    args = parser.parse_args()

    print("="*60)
    print("CNN CLASSIFIER TRAINING")
    print("="*60)
    print(f"Architecture: {args.model}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print("="*60)
    
    classifier = CNNClassifier(model_type=args.model)
    
    total_params = sum(p.numel() for p in classifier.model.parameters())
    trainable_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"📐 Toplam parametre: {total_params:,}")
    print(f"📐 Eğitilebilir parametre: {trainable_params:,}")
    print("="*60)
    
    classifier.prepare_data(args.dataset, batch_size=args.batch_size)
    
    best_model_path = classifier.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    classifier.load(best_model_path)
    results = classifier.evaluate()
    
    print("\n" + "="*60)
    print("✅ TRAINING TAMAMLANDI")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
