import cv2
from pathlib import Path
from src.models.ensemble_classifier import EnsembleClassifier
from src.models.feature_classifier import FeatureBasedClassifier
from src.models.cnn_classifier import CNNClassifier
import json

def test_yolo_simulation(test_dir, feature_model_path, cnn_model_path):
    """
    YOLO simülasyonu - YOLO'nun tahminlerini test eder.
    
    Args:
        test_dir: Test klasörü (helmet ve no_helmet alt klasörleri içeren)
        feature_model_path: Feature-based model yolu
        cnn_model_path: CNN model yolu
    """
    test_dir = Path(test_dir)
    
    print("="*70)
    print(" "*20 + "YOLO SİMÜLASYON TESTİ")
    print("="*70)
    print("\nSenaryo: YOLO'nun tahminlerini simüle ediyoruz")
    print("Klasör adı = YOLO'nun tahmini")
    print("Bakalım modelimiz ne kadar düzeltme yapacak!\n")
    print("="*70)
    
    print("\n📦 Modeller yükleniyor...")
    ensemble = EnsembleClassifier(
        feature_model_path=feature_model_path,
        cnn_model_path=cnn_model_path,
        feature_weight=0.4,
        cnn_weight=0.6
    )
    
    class_names = {0: 'no_helmet', 1: 'helmet'}
    
    results = {
        'helmet': {'images': [], 'yolo_correct': 0, 'yolo_wrong': 0, 'corrected': 0, 'still_wrong': 0},
        'no_helmet': {'images': [], 'yolo_correct': 0, 'yolo_wrong': 0, 'corrected': 0, 'still_wrong': 0}
    }
    
    for yolo_class_name in ['helmet', 'no_helmet']:
        yolo_class_id = 1 if yolo_class_name == 'helmet' else 0
        class_dir = test_dir / yolo_class_name
        
        if not class_dir.exists():
            print(f"⚠️  {class_dir} bulunamadı!")
            continue
        
        print(f"\n{'='*70}")
        print(f"📁 YOLO TAHMİNİ: {yolo_class_name.upper()}")
        print(f"{'='*70}")
        
        images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        
        for img_path in images:
            result = ensemble.predict(img_path, return_details=True)
            
            our_prediction = result['prediction']
            our_class_name = class_names[our_prediction]
            confidence = result['confidence']
            
            feature_pred = result['feature_classifier']['prediction']
            feature_conf = result['feature_classifier']['confidence']
            cnn_pred = result['cnn_classifier']['prediction']
            cnn_conf = result['cnn_classifier']['confidence']
            
            agreement = '✅' if our_prediction == yolo_class_id else '❌'
            
            print(f"\n📷 {img_path.name}")
            print(f"  YOLO tahmini: {yolo_class_name}")
            print(f"  Bizim tahmin: {our_class_name} (confidence: {confidence:.2%}) {agreement}")
            print(f"    ├─ Feature-based: {class_names[feature_pred]} ({feature_conf:.2%})")
            print(f"    └─ CNN: {class_names[cnn_pred]} ({cnn_conf:.2%})")
            
            img_result = {
                'filename': img_path.name,
                'yolo_prediction': yolo_class_name,
                'our_prediction': our_class_name,
                'confidence': confidence,
                'feature_prediction': class_names[feature_pred],
                'feature_confidence': feature_conf,
                'cnn_prediction': class_names[cnn_pred],
                'cnn_confidence': cnn_conf,
                'agreement': our_prediction == yolo_class_id
            }
            
            results[yolo_class_name]['images'].append(img_result)
            
            if our_prediction == yolo_class_id:
                results[yolo_class_name]['yolo_correct'] += 1
                print(f"  ✅ Modelimiz YOLO ile AYNI FİKİRDE")
            else:
                results[yolo_class_name]['yolo_wrong'] += 1
                print(f"  ⚠️  Modelimiz YOLO'yu DÜZELTİYOR!")
                print(f"     YOLO: {yolo_class_name} → Bizim: {our_class_name}")
    
    print("\n" + "="*70)
    print(" "*25 + "📊 ÖZET RAPOR")
    print("="*70)
    
    total_images = 0
    total_agreements = 0
    total_corrections = 0
    
    for yolo_class in ['helmet', 'no_helmet']:
        data = results[yolo_class]
        total = len(data['images'])
        agreements = data['yolo_correct']
        corrections = data['yolo_wrong']
        
        total_images += total
        total_agreements += agreements
        total_corrections += corrections
        
        print(f"\n🏷️  YOLO TAHMİNİ: {yolo_class.upper()}")
        print(f"  Toplam görüntü: {total}")
        print(f"  Modelimiz aynı fikirde: {agreements} ({100*agreements/total if total > 0 else 0:.1f}%)")
        print(f"  Modelimiz düzeltti: {corrections} ({100*corrections/total if total > 0 else 0:.1f}%)")
    
    print("\n" + "-"*70)
    print(f"\n🎯 GENEL SONUÇ:")
    print(f"  Toplam test görüntüsü: {total_images}")
    print(f"  YOLO ile anlaşma: {total_agreements}/{total_images} ({100*total_agreements/total_images if total_images > 0 else 0:.1f}%)")
    print(f"  Düzeltme yapılan: {total_corrections}/{total_images} ({100*total_corrections/total_images if total_images > 0 else 0:.1f}%)")
    
    if total_corrections > 0:
        print(f"\n💡 Modelimiz {total_corrections} görüntüde YOLO'dan farklı düşünüyor!")
        print(f"   Bu, post-processing'in işe yaradığını gösterir.")
    else:
        print(f"\n✅ Modelimiz tüm görüntülerde YOLO ile aynı fikirde!")
    
    print("\n" + "="*70)
    
    output_file = 'yolo_simulation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Detaylı sonuçlar kaydedildi: {output_file}")
    
    return results

def visualize_results(test_dir, results):
    """
    Sonuçları görselleştirir - her görüntüyü tahminlerle birlikte gösterir.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    test_dir = Path(test_dir)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('YOLO Simülasyon Testi - Tahminler', fontsize=16, fontweight='bold')
    
    row_idx = 0
    col_idx = 0
    
    for yolo_class in ['helmet', 'no_helmet']:
        class_dir = test_dir / yolo_class
        
        for img_result in results[yolo_class]['images']:
            img_path = class_dir / img_result['filename']
            
            if not img_path.exists():
                continue
            
            img = Image.open(img_path)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')
            
            yolo_pred = img_result['yolo_prediction']
            our_pred = img_result['our_prediction']
            conf = img_result['confidence']
            agreement = img_result['agreement']
            
            if agreement:
                title_color = 'green'
                status = '✅ Anlaşma'
            else:
                title_color = 'red'
                status = '⚠️ Düzeltme'
            
            title = f"YOLO: {yolo_pred}\nBiz: {our_pred} ({conf:.0%})\n{status}"
            ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')
            
            col_idx += 1
            if col_idx >= 5:
                col_idx = 0
                row_idx += 1
    
    plt.tight_layout()
    plt.savefig('yolo_simulation_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Görselleştirme kaydedildi: yolo_simulation_visualization.png")
    plt.close()

if __name__ == "__main__":
    test_dir = 'test/cnn_test'
    feature_model_path = 'models/trained/feature_classifier_xgboost.pkl'
    cnn_model_path = 'models/trained/cnn_classifier_best.pth'
    
    results = test_yolo_simulation(test_dir, feature_model_path, cnn_model_path)
    
    print("\n📊 Görselleştirme oluşturuluyor...")
    visualize_results(test_dir, results)
    
    print("\n✅ Test tamamlandı!")
