#!/usr/bin/env python3
"""
Master script - Training pipeline'ını çalıştırır.

Adımlar:
  1. Dataset hazırlama (train/val/test split)
  2. CNN classifier eğitimi (HelmetClassifierNet)
"""

import sys
from pathlib import Path

def main():
    print("="*70)
    print(" "*15 + "HELMET CLASSIFIER TRAINING PIPELINE")
    print("="*70)
    print("\nBu script sırasıyla şunları yapacak:")
    print("  1. Dataset hazırlama (train/val/test split)")
    print("  2. CNN classifier eğitimi (HelmetClassifierNet)")
    print("="*70)
    
    response = input("\nDevam etmek istiyor musunuz? (y/n): ").lower()
    if response != 'y':
        print("❌ İşlem iptal edildi.")
        return
    
    # ── ADIM 1: Dataset Hazırlama ────────────────────────────────────────────
    print("\n" + "="*70)
    print("ADIM 1/2: Dataset Hazırlama")
    print("="*70)
    
    dataset_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    try:
        from src.data_preparation.prepare_dataset import prepare_dataset
        
        source_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images'
        
        stats = prepare_dataset(
            source_dir=source_dir,
            output_dir=dataset_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        print("\n✅ Dataset hazırlama tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("Dataset zaten hazır olabilir, devam ediliyor...")
    
    # ── ADIM 2: CNN Classifier Eğitimi ───────────────────────────────────────
    print("\n" + "="*70)
    print("ADIM 2/2: CNN Classifier Eğitimi")
    print("="*70)
    
    try:
        from src.models.cnn_classifier import CNNClassifier
        
        classifier = CNNClassifier(model_type='efficientnet')
        classifier.prepare_data(dataset_dir, batch_size=16)
        best_model_path = classifier.train(
            num_epochs=30,
            learning_rate=0.001,
            output_dir='models/trained',
            model_name='cnn_classifier_best.pth'
        )
        classifier.load(best_model_path)
        results = classifier.evaluate()
        
        print(f"\n✅ CNN classifier eğitimi tamamlandı! (Accuracy: {results['accuracy']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✅ TÜM PIPELINE TAMAMLANDI!")
    print("="*70)
    print("\nOluşturulan dosyalar:")
    print(f"  • {best_model_path} - CNN model")
    print("  • *_confusion_matrix.png - Confusion matrix grafikleri")
    print("  • cnn_classifier_training_history.png - Training history")
    print("\nKullanım:")
    print("  python pipeline_test.py  # YOLO + CNN pipeline testi için")
    print("="*70)

if __name__ == "__main__":
    main()
