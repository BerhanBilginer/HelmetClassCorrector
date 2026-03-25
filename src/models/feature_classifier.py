"""
⚠️  DEPRECATED — Bu modül artık kullanılmıyor.

Feature-based classifier (XGBoost) kaldırıldı. Inference ve eğitimde
sadece CNN (HelmetClassifierNet) kullanılıyor.
Bkz: src/models/cnn_classifier.py
"""

import warnings
warnings.warn(
    "feature_classifier.py is deprecated. Use cnn_classifier.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExtractor:
    """
    Görüntüden parlaklık, keskinlik ve doygunluk özelliklerini çıkarır.
    """
    @staticmethod
    def extract_features(image_path):
        """
        Tek bir görüntüden özellikleri çıkarır.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        brightness = img_gray.mean()
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        saturation = img_hsv[:, :, 1].mean()
        
        return {
            'brightness': brightness,
            'sharpness': sharpness,
            'saturation': saturation
        }
    
    @staticmethod
    def extract_from_directory(directory, label):
        """
        Bir klasördeki tüm görüntülerden özellikleri çıkarır.
        """
        directory = Path(directory)
        features_list = []
        
        for img_path in directory.glob('*.png'):
            features = FeatureExtractor.extract_features(img_path)
            if features:
                features['label'] = label
                features['filename'] = img_path.name
                features_list.append(features)
        
        for img_path in directory.glob('*.jpg'):
            features = FeatureExtractor.extract_features(img_path)
            if features:
                features['label'] = label
                features['filename'] = img_path.name
                features_list.append(features)
        
        return features_list

class FeatureBasedClassifier:
    """
    Parlaklık, keskinlik ve doygunluk özelliklerine dayalı classifier.
    """
    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: 'xgboost' veya 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = ['brightness', 'sharpness', 'saturation']
        
        if model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")
    
    def prepare_data(self, dataset_dir):
        """
        Dataset'ten özellikleri çıkarır ve DataFrame oluşturur.
        """
        dataset_dir = Path(dataset_dir)
        
        print("📊 Özellikler çıkarılıyor...")
        
        train_features = []
        train_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'train' / 'helmet', label=1
        ))
        train_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'train' / 'no_helmet', label=0
        ))
        
        val_features = []
        val_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'val' / 'helmet', label=1
        ))
        val_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'val' / 'no_helmet', label=0
        ))
        
        test_features = []
        test_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'test' / 'helmet', label=1
        ))
        test_features.extend(FeatureExtractor.extract_from_directory(
            dataset_dir / 'test' / 'no_helmet', label=0
        ))
        
        train_df = pd.DataFrame(train_features)
        val_df = pd.DataFrame(val_features)
        test_df = pd.DataFrame(test_features)
        
        print(f"✓ Train: {len(train_df)} samples")
        print(f"✓ Val: {len(val_df)} samples")
        print(f"✓ Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def train(self, train_df, val_df=None):
        """
        Modeli eğitir.
        """
        X_train = train_df[self.feature_names].values
        y_train = train_df['label'].values
        
        print(f"\n🎯 {self.model_type.upper()} modeli eğitiliyor...")
        
        if val_df is not None and self.model_type == 'xgboost':
            X_val = val_df[self.feature_names].values
            y_val = val_df['label'].values
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        print(f"✓ Training Accuracy: {train_acc:.4f}")
        
        if val_df is not None:
            X_val = val_df[self.feature_names].values
            y_val = val_df['label'].values
            val_acc = self.model.score(X_val, y_val)
            print(f"✓ Validation Accuracy: {val_acc:.4f}")
    
    def evaluate(self, test_df):
        """
        Test seti üzerinde değerlendirme yapar.
        """
        X_test = test_df[self.feature_names].values
        y_test = test_df['label'].values
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        print("\n" + "="*60)
        print("📊 TEST SONUÇLARI")
        print("="*60)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['no_helmet', 'helmet']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        self.plot_confusion_matrix(cm)
        self.plot_feature_importance()
        
        return {
            'accuracy': acc,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm):
        """
        Confusion matrix görselleştirir.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['no_helmet', 'helmet'],
                    yticklabels=['no_helmet', 'helmet'])
        plt.title(f'{self.model_type.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'feature_classifier_{self.model_type}_confusion_matrix.png', dpi=150)
        print(f"\n✓ Confusion matrix kaydedildi: feature_classifier_{self.model_type}_confusion_matrix.png")
        plt.close()
    
    def plot_feature_importance(self):
        """
        Feature importance görselleştirir.
        """
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        else:
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.feature_names, importance)
        plt.title(f'{self.model_type.upper()} - Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'feature_classifier_{self.model_type}_importance.png', dpi=150)
        print(f"✓ Feature importance kaydedildi: feature_classifier_{self.model_type}_importance.png")
        plt.close()
    
    def predict(self, image_path, return_proba=False):
        """
        Tek bir görüntü için tahmin yapar.
        """
        features = FeatureExtractor.extract_features(image_path)
        if features is None:
            return None
        
        X = np.array([[features['brightness'], features['sharpness'], features['saturation']]])
        
        if return_proba:
            proba = self.model.predict_proba(X)[0]
            return {
                'prediction': int(self.model.predict(X)[0]),
                'confidence': float(max(proba)),
                'probabilities': {'no_helmet': float(proba[0]), 'helmet': float(proba[1])}
            }
        else:
            return int(self.model.predict(X)[0])
    
    def save(self, filepath):
        """
        Modeli kaydeder.
        """
        joblib.dump(self.model, filepath)
        print(f"✓ Model kaydedildi: {filepath}")
    
    def load(self, filepath):
        """
        Modeli yükler.
        """
        self.model = joblib.load(filepath)
        print(f"✓ Model yüklendi: {filepath}")

def main():
    dataset_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    print("="*60)
    print("FEATURE-BASED CLASSIFIER TRAINING")
    print("="*60)
    print("Features: Brightness, Sharpness, Saturation")
    print("="*60)
    
    classifier = FeatureBasedClassifier(model_type='xgboost')
    
    train_df, val_df, test_df = classifier.prepare_data(dataset_dir)
    
    classifier.train(train_df, val_df)
    
    results = classifier.evaluate(test_df)
    
    classifier.save('feature_classifier_xgboost.pkl')
    
    print("\n" + "="*60)
    print("✅ TRAINING TAMAMLANDI")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
