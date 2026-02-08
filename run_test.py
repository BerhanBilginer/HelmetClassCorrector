#!/usr/bin/env python3
"""
Quick test script - Trained modelleri test eder.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.test_yolo_simulation import test_yolo_simulation, visualize_results

def main():
    print("="*70)
    print(" "*20 + "YOLO SİMÜLASYON TESTİ")
    print("="*70)
    
    test_dir = 'test/cnn_test'
    feature_model_path = 'models/trained/feature_classifier_xgboost.pkl'
    cnn_model_path = 'models/trained/cnn_classifier_best.pth'
    
    # Check if models exist
    if not Path(feature_model_path).exists():
        print(f"❌ Model bulunamadı: {feature_model_path}")
        print("Önce modelleri eğitmelisiniz: python train_all.py")
        return
    
    if not Path(cnn_model_path).exists():
        print(f"❌ Model bulunamadı: {cnn_model_path}")
        print("Önce modelleri eğitmelisiniz: python train_all.py")
        return
    
    print("\n🚀 Test başlıyor...\n")
    
    results = test_yolo_simulation(test_dir, feature_model_path, cnn_model_path)
    
    print("\n📊 Görselleştirme oluşturuluyor...")
    visualize_results(test_dir, results)
    
    print("\n✅ Test tamamlandı!")
    print(f"Sonuçlar: results/reports/yolo_simulation_results.json")
    print(f"Görsel: results/visualizations/yolo_simulation_visualization.png")

if __name__ == "__main__":
    main()
