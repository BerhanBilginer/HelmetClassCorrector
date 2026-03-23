#!/usr/bin/env python3
"""
Master script - Tüm training pipeline'ını çalıştırır.
"""

import sys
from pathlib import Path

def main():
    print("="*70)
    print(" "*15 + "HELMET CLASSIFIER TRAINING PIPELINE")
    print("="*70)
    print("\nBu script sırasıyla şunları yapacak:")
    print("  1. Dataset hazırlama (train/val/test split)")
    print("  2. Feature-based classifier eğitimi (XGBoost)")
    print("  3. CNN classifier eğitimi")
    print("  4. Ensemble değerlendirmesi")
    print("="*70)
    
    response = input("\nDevam etmek istiyor musunuz? (y/n): ").lower()
    if response != 'y':
        print("❌ İşlem iptal edildi.")
        return
    
    print("\n" + "="*70)
    print("ADIM 1/4: Dataset Hazırlama")
    print("="*70)
    
    try:
        from src.data_preparation.prepare_dataset import prepare_dataset
        
        source_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images'
        output_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
        
        stats = prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        print("\n✅ Dataset hazırlama tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("Dataset zaten hazır olabilir, devam ediliyor...")
    
    print("\n" + "="*70)
    print("ADIM 2/4: Feature-Based Classifier Eğitimi")
    print("="*70)
    
    try:
        from src.models.feature_classifier import FeatureBasedClassifier
        
        dataset_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
        
        classifier = FeatureBasedClassifier(model_type='xgboost')
        train_df, val_df, test_df = classifier.prepare_data(dataset_dir)
        classifier.train(train_df, val_df)
        results = classifier.evaluate(test_df)
        classifier.save('models/trained/feature_classifier_xgboost.pkl')
        
        print(f"\n✅ Feature classifier eğitimi tamamlandı! (Accuracy: {results['accuracy']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("ADIM 3/4: CNN Classifier Eğitimi")
    print("="*70)
    
    try:
        from src.models.cnn_classifier import CNNClassifier
        
        classifier = CNNClassifier(model_type='efficientnet')
        classifier.prepare_data(dataset_dir, batch_size=16)
        classifier.train(num_epochs=30, learning_rate=0.001)
        classifier.load('cnn_classifier_best.pth')
        results = classifier.evaluate()
        
        print(f"\n✅ CNN classifier eğitimi tamamlandı! (Accuracy: {results['accuracy']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("ADIM 4/4: Ensemble Değerlendirmesi")
    print("="*70)
    
    try:
        from src.models.ensemble_classifier import EnsembleClassifier
        
        ensemble = EnsembleClassifier(
            feature_model_path='models/trained/feature_classifier_xgboost.pkl',
            cnn_model_path='models/trained/cnn_classifier_best.pth',
            feature_weight=0.4,
            cnn_weight=0.6
        )
        
        results = ensemble.evaluate_on_dataset(dataset_dir, split='test')
        ensemble.save_config('config/ensemble_config.json')
        
        print(f"\n✅ Ensemble değerlendirmesi tamamlandı! (Accuracy: {results['accuracy']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✅ TÜM PIPELINE TAMAMLANDI!")
    print("="*70)
    print("\nOluşturulan dosyalar:")
    print("  • feature_classifier_xgboost.pkl - Feature-based model")
    print("  • cnn_classifier_best.pth - CNN model")
    print("  • ensemble_config.json - Ensemble konfigürasyonu")
    print("  • *_confusion_matrix.png - Confusion matrix grafikleri")
    print("  • *_importance.png - Feature importance grafikleri")
    print("  • cnn_classifier_training_history.png - Training history")
    print("\nKullanım:")
    print("  python yolo_pipeline.py  # YOLO entegrasyonu için")
    print("="*70)

if __name__ == "__main__":
    main()
