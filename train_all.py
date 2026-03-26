#!/usr/bin/env python3
"""
CNN Classifier Training Script

HelmetClassifierNet (EfficientNet-B0 + CBAM + FPN + Color Branch) eğitimi.
2-fazlı transfer learning: Phase 1 (frozen backbone) + Phase 2 (full fine-tune)
"""

import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train HelmetClassifierNet (EfficientNet-B0 + CBAM + FPN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset paths
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Dataset klasörü (train/val/test içeren)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Toplam epoch sayısı'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    # Output paths
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='Model ve grafiklerin kaydedileceği klasör'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='cnn_classifier_best.pth',
        help='Kaydedilecek model dosya adı'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print(" "*15 + "HELMET CLASSIFIER TRAINING")
    print("="*70)
    print(f"\n📁 Dataset:       {args.dataset_dir}")
    print(f"📊 Epochs:        {args.epochs}")
    print(f"📦 Batch size:    {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"💾 Output dir:    {args.output_dir}")
    print(f"🏷️  Model name:    {args.model_name}")
    print("="*70)
    
    # Import CNN classifier
    try:
        from src.models.cnn_classifier import CNNClassifier
    except ImportError as e:
        print(f"\n❌ Import hatası: {e}")
        print("src/models/cnn_classifier.py dosyasını kontrol edin.")
        return 1
    
    # Check dataset exists
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"\n❌ Hata: Dataset klasörü bulunamadı: {dataset_path}")
        print("\n💡 Dataset hazırlamak için:")
        print("   python src/data_preparation/crop_augmentor.py  # Class balancing")
        return 1
    
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    if not train_dir.exists() or not val_dir.exists():
        print(f"\n❌ Hata: train/ veya val/ klasörü bulunamadı")
        print(f"Beklenen yapı: {dataset_path}/train/helmet, {dataset_path}/train/no_helmet")
        return 1
    
    # Initialize classifier
    print("\n🔧 Model oluşturuluyor...")
    classifier = CNNClassifier(model_type='efficientnet')
    
    # Prepare data
    print(f"\n📚 Dataset yükleniyor: {args.dataset_dir}")
    classifier.prepare_data(args.dataset_dir, batch_size=args.batch_size)
    
    # Train
    print("\n🚀 Eğitim başlatılıyor...")
    print("="*70)
    
    try:
        best_model_path = classifier.train(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"\n❌ Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Evaluate
    print("\n📊 Model değerlendiriliyor...")
    classifier.load(best_model_path)
    results = classifier.evaluate()
    
    # Summary
    output_dir = Path(args.output_dir)
    print("\n" + "="*70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("="*70)
    print(f"\n📈 Test Accuracy: {results['accuracy']:.2%}")
    print(f"\n💾 Oluşturulan dosyalar:")
    print(f"  • {best_model_path}")
    print(f"  • {output_dir / 'cnn_classifier_confusion_matrix.png'}")
    print(f"  • {output_dir / 'cnn_classifier_training_history.png'}")
    print("\n🚀 Kullanım:")
    print("  python pipeline_test.py")
    print("  streamlit run streamlit_pipeline.py")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    main()
