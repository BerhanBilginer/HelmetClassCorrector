#!/usr/bin/env python3
"""
Helmet Classifier v5 Training Script

Kullanım örnekleri:

  # Varsayılan ayarlar (önerilen başlangıç)
  python train_all.py --dataset-dir dataset

  # 384px input (daha iyi detay, batch-size düşür)
  python train_all.py --dataset-dir dataset --img-size 384 --batch-size 8

  # Sadece Focal Loss dene (mixup kapalı)
  python train_all.py --dataset-dir dataset --mixup-alpha 0 --cutmix-alpha 0

  # Edge-texture branch yerine color branch (v4 karşılaştırma)
  python train_all.py --dataset-dir dataset --branch color --model-name v5_color.pth

  # Colab crash sonrası kaldığı yerden devam
  python train_all.py --dataset-dir dataset --resume

  # Agresif ayarlar (GPU bellek yeterliyse)
  python train_all.py --dataset-dir dataset --img-size 384 --batch-size 8 \
                      --epochs 40 --lr 0.0008 --label-smoothing 0.15
"""

import sys
import argparse
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train HelmetClassifierNet v5 (EfficientNet-B0 + CBAM + FPN + EdgeTexture)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument('--dataset-dir', type=str, default='dataset')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)

    # Model architecture
    parser.add_argument('--img-size', type=int, default=224, choices=[224, 384],
                        help='Input resolution: 224 (standard) veya 384 (high-detail)')
    parser.add_argument('--branch', type=str, default='edge_texture',
                        choices=['edge_texture', 'color'],
                        help='Side branch tipi: edge_texture (v5) veya color (v4 legacy)')
    parser.add_argument('--disable-center-guidance', action='store_true',
                        help='Merkez odaklı guided attention/pooling modülünü kapat')
    parser.add_argument('--center-guidance-strength', type=float, default=0.35,
                        help='Merkez prior etkisi (yüksek = merkez daha baskın)')
    parser.add_argument('--center-guidance-sigma', type=float, default=None,
                        help='Legacy isotropic sigma. Verilirse sigma_x/sigma_y yerine kullanılır.')
    parser.add_argument('--center-guidance-sigma-x', type=float, default=DEFAULT_CENTER_GUIDANCE_SIGMA_X,
                        help='Merkez prior yatay genişliği')
    parser.add_argument('--center-guidance-sigma-y', type=float, default=DEFAULT_CENTER_GUIDANCE_SIGMA_Y,
                        help='Merkez prior dikey genişliği')
    parser.add_argument('--center-guidance-center-x', type=float, default=DEFAULT_CENTER_GUIDANCE_CENTER_X,
                        help='Merkez prior yatay kaydırma (-1 sol, +1 sağ)')
    parser.add_argument('--center-guidance-center-y', type=float, default=DEFAULT_CENTER_GUIDANCE_CENTER_Y,
                        help='Merkez prior dikey kaydırma (-1 yukarı, +1 aşağı)')

    # Loss & regularization
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce'],
                        help='Loss fonksiyonu: focal (zor örneklere odak) veya ce (standart)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing parametresi (0=kapalı, 0.1=önerilen)')
    parser.add_argument('--mixup-alpha', type=float, default=0.4,
                        help='Mixup alpha (0=kapalı)')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                        help='CutMix alpha (0=kapalı)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Her batch\'te mix uygulanma olasılığı')
    parser.add_argument('--helmet-focus-loss-weight', type=float, default=DEFAULT_HELMET_FOCUS_LOSS_WEIGHT,
                        help='Helmet örneklerinde üst-kask bölgesini teşvik eden yardımcı loss ağırlığı')
    parser.add_argument('--helmet-focus-sigma-x', type=float, default=DEFAULT_HELMET_FOCUS_SIGMA_X,
                        help='Helmet focus prior yatay genişliği')
    parser.add_argument('--helmet-focus-sigma-y', type=float, default=DEFAULT_HELMET_FOCUS_SIGMA_Y,
                        help='Helmet focus prior dikey genişliği')
    parser.add_argument('--helmet-focus-center-x', type=float, default=DEFAULT_HELMET_FOCUS_CENTER_X,
                        help='Helmet focus prior yatay kaydırma')
    parser.add_argument('--helmet-focus-center-y', type=float, default=DEFAULT_HELMET_FOCUS_CENTER_Y,
                        help='Helmet focus prior dikey kaydırma')
    parser.add_argument('--helmet-focus-outside-penalty', type=float, default=DEFAULT_HELMET_FOCUS_OUTSIDE_PENALTY,
                        help='Prior dışına kayan aktivasyonlara uygulanan ek ceza')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/trained')
    parser.add_argument('--model-name', type=str, default='helmet_classifier_v5.pth')

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Checkpoint\'tan kaldığı yerden devam et')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" " * 10 + "HELMET CLASSIFIER v5 TRAINING")
    print("=" * 70)
    print(f"\n📁 Dataset:         {args.dataset_dir}")
    print(f"📐 Input size:      {args.img_size}×{args.img_size}")
    print(f"🧠 Branch:          {args.branch}")
    print(f"🎯 Center guidance: {'OFF' if args.disable_center_guidance else 'ON'} "
          f"(strength={args.center_guidance_strength}, sigma_x={args.center_guidance_sigma_x}, "
          f"sigma_y={args.center_guidance_sigma_y}, center=({args.center_guidance_center_x}, "
          f"{args.center_guidance_center_y}))")
    print(f"📊 Epochs:          {args.epochs}")
    print(f"📦 Batch size:      {args.batch_size}")
    print(f"📈 Learning rate:   {args.lr}")
    print(f"🎯 Loss:            {args.loss} (smoothing={args.label_smoothing})")
    print(f"🔀 Mixup α={args.mixup_alpha} | CutMix α={args.cutmix_alpha} | prob={args.mixup_prob}")
    print(f"🪖 Helmet focus:    weight={args.helmet_focus_loss_weight} "
          f"(sigma_x={args.helmet_focus_sigma_x}, sigma_y={args.helmet_focus_sigma_y}, "
          f"center=({args.helmet_focus_center_x}, {args.helmet_focus_center_y}))")
    print(f"💾 Output:          {args.output_dir}/{args.model_name}")
    print("=" * 70)

    # Import
    try:
        from src.models.cnn_classifier import CNNClassifier
    except ImportError as e:
        print(f"\n❌ Import hatası: {e}")
        print("src/models/cnn_classifier.py dosyasını kontrol edin.")
        return 1

    # Dataset kontrolü
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"\n❌ Dataset bulunamadı: {dataset_path}")
        return 1

    for subdir in ['train/helmet', 'train/no_helmet', 'val/helmet', 'val/no_helmet']:
        if not (dataset_path / subdir).exists():
            print(f"\n❌ Eksik klasör: {dataset_path / subdir}")
            return 1

    # Classifier oluştur
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
    print(f"\n📐 Total: {total_params:,} | Trainable (Phase 1): {trainable:,}")

    # Data
    classifier.prepare_data(args.dataset_dir, batch_size=args.batch_size)

    # Train
    if args.resume:
        print("\n🔄 Checkpoint'tan devam ediliyor...")
    else:
        print("\n🚀 Eğitim başlatılıyor...")
    best_model_path = classifier.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        model_name=args.model_name,
        resume=args.resume,
    )

    # Evaluate
    print("\n📊 Final değerlendirme...")
    classifier.load(best_model_path)
    results = classifier.evaluate(model_path=best_model_path)

    # Summary
    model_dir = Path(best_model_path).parent
    print(f"\n{'='*70}")
    print(f"✅ EĞİTİM TAMAMLANDI!")
    print(f"{'='*70}")
    print(f"\n📈 Test Accuracy: {results['accuracy']:.2%}")
    print(f"\n💾 Dosyalar:")
    print(f"  • {best_model_path}")
    print(f"  • {model_dir / 'checkpoint_last.pth'}")
    print(f"  • {model_dir / 'training_history.png'}")
    print(f"  • {model_dir / 'confusion_matrix.png'}")
    print(f"\n🔬 Sonraki adım — Grad-CAM analizi:")
    print(f"  python grad_cam_analysis.py --model {best_model_path} --dataset {args.dataset_dir} --tta")
    print(f"\n🚀 Pipeline testi:")
    print(f"  python pipeline_test.py")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
