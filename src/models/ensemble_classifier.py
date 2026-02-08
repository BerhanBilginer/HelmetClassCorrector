import numpy as np
from pathlib import Path
import joblib
import torch
from .feature_classifier import FeatureBasedClassifier, FeatureExtractor
from .cnn_classifier import CNNClassifier
import json

class EnsembleClassifier:
    """
    Feature-based ve CNN classifier'ları birleştiren ensemble sistem.
    """
    def __init__(self, feature_model_path=None, cnn_model_path=None, 
                 feature_weight=0.4, cnn_weight=0.6):
        """
        Args:
            feature_model_path: Feature-based model dosya yolu
            cnn_model_path: CNN model dosya yolu
            feature_weight: Feature-based model ağırlığı
            cnn_weight: CNN model ağırlığı
        """
        self.feature_weight = feature_weight
        self.cnn_weight = cnn_weight
        
        self.feature_classifier = None
        self.cnn_classifier = None
        
        if feature_model_path:
            self.load_feature_model(feature_model_path)
        
        if cnn_model_path:
            self.load_cnn_model(cnn_model_path)
    
    def load_feature_model(self, model_path):
        """
        Feature-based modeli yükler.
        """
        self.feature_classifier = FeatureBasedClassifier(model_type='xgboost')
        self.feature_classifier.load(model_path)
        print(f"✓ Feature model yüklendi: {model_path}")
    
    def load_cnn_model(self, model_path):
        """
        CNN modeli yükler.
        """
        self.cnn_classifier = CNNClassifier()
        self.cnn_classifier.load(model_path)
        print(f"✓ CNN model yüklendi: {model_path}")
    
    def predict(self, image_path, return_details=False):
        """
        Ensemble prediction yapar.
        
        Args:
            image_path: Görüntü dosya yolu
            return_details: True ise detaylı sonuç döner
        
        Returns:
            prediction: 0 (no_helmet) veya 1 (helmet)
            veya detaylı sonuç dict
        """
        if self.feature_classifier is None or self.cnn_classifier is None:
            raise ValueError("Her iki model de yüklenmiş olmalı!")
        
        feature_result = self.feature_classifier.predict(image_path, return_proba=True)
        cnn_result = self.cnn_classifier.predict(image_path, return_proba=True)
        
        feature_proba = feature_result['probabilities']['helmet']
        cnn_proba = cnn_result['probabilities']['helmet']
        
        weighted_proba = (self.feature_weight * feature_proba + 
                         self.cnn_weight * cnn_proba)
        
        final_prediction = 1 if weighted_proba >= 0.5 else 0
        
        if return_details:
            return {
                'prediction': final_prediction,
                'confidence': float(max(weighted_proba, 1 - weighted_proba)),
                'weighted_proba_helmet': float(weighted_proba),
                'feature_classifier': {
                    'prediction': feature_result['prediction'],
                    'confidence': feature_result['confidence'],
                    'proba_helmet': feature_proba
                },
                'cnn_classifier': {
                    'prediction': cnn_result['prediction'],
                    'confidence': cnn_result['confidence'],
                    'proba_helmet': cnn_proba
                },
                'weights': {
                    'feature': self.feature_weight,
                    'cnn': self.cnn_weight
                }
            }
        else:
            return final_prediction
    
    def predict_batch(self, image_paths):
        """
        Birden fazla görüntü için tahmin yapar.
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_details=True)
            result['image_path'] = str(img_path)
            results.append(result)
        return results
    
    def evaluate_on_dataset(self, dataset_dir, split='test'):
        """
        Dataset üzerinde değerlendirme yapar.
        """
        dataset_dir = Path(dataset_dir) / split
        
        helmet_dir = dataset_dir / 'helmet'
        no_helmet_dir = dataset_dir / 'no_helmet'
        
        print(f"\n📊 {split.upper()} seti değerlendiriliyor...")
        
        correct = 0
        total = 0
        
        results = {
            'helmet': {'correct': 0, 'total': 0, 'predictions': []},
            'no_helmet': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        for img_path in helmet_dir.glob('*.png'):
            pred = self.predict(img_path)
            results['helmet']['predictions'].append(pred)
            results['helmet']['total'] += 1
            total += 1
            if pred == 1:
                results['helmet']['correct'] += 1
                correct += 1
        
        for img_path in helmet_dir.glob('*.jpg'):
            pred = self.predict(img_path)
            results['helmet']['predictions'].append(pred)
            results['helmet']['total'] += 1
            total += 1
            if pred == 1:
                results['helmet']['correct'] += 1
                correct += 1
        
        for img_path in no_helmet_dir.glob('*.png'):
            pred = self.predict(img_path)
            results['no_helmet']['predictions'].append(pred)
            results['no_helmet']['total'] += 1
            total += 1
            if pred == 0:
                results['no_helmet']['correct'] += 1
                correct += 1
        
        for img_path in no_helmet_dir.glob('*.jpg'):
            pred = self.predict(img_path)
            results['no_helmet']['predictions'].append(pred)
            results['no_helmet']['total'] += 1
            total += 1
            if pred == 0:
                results['no_helmet']['correct'] += 1
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print("\n" + "="*60)
        print("📊 ENSEMBLE SONUÇLARI")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"\nHelmet:")
        print(f"  Correct: {results['helmet']['correct']}/{results['helmet']['total']}")
        print(f"  Accuracy: {results['helmet']['correct']/results['helmet']['total']:.4f}")
        print(f"\nNo Helmet:")
        print(f"  Correct: {results['no_helmet']['correct']}/{results['no_helmet']['total']}")
        print(f"  Accuracy: {results['no_helmet']['correct']/results['no_helmet']['total']:.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'results': results
        }
    
    def save_config(self, filepath='ensemble_config.json'):
        """
        Ensemble konfigürasyonunu kaydeder.
        """
        config = {
            'feature_weight': self.feature_weight,
            'cnn_weight': self.cnn_weight
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Config kaydedildi: {filepath}")
    
    @classmethod
    def load_from_config(cls, config_path, feature_model_path, cnn_model_path):
        """
        Config dosyasından ensemble oluşturur.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            feature_model_path=feature_model_path,
            cnn_model_path=cnn_model_path,
            feature_weight=config['feature_weight'],
            cnn_weight=config['cnn_weight']
        )

def main():
    print("="*60)
    print("ENSEMBLE CLASSIFIER EVALUATION")
    print("="*60)
    
    ensemble = EnsembleClassifier(
        feature_model_path='feature_classifier_xgboost.pkl',
        cnn_model_path='cnn_classifier_best.pth',
        feature_weight=0.4,
        cnn_weight=0.6
    )
    
    dataset_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    results = ensemble.evaluate_on_dataset(dataset_dir, split='test')
    
    ensemble.save_config('ensemble_config.json')
    
    print("\n✅ Değerlendirme tamamlandı!")
    print(f"Final Ensemble Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
