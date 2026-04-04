import cv2
import numpy as np
from pathlib import Path
from src.models.ensemble_classifier import EnsembleClassifier
from src.utils.image_ops import DEFAULT_CONTEXT_CROP_CONFIG, crop_with_context
import json
from datetime import datetime

class YOLOPostProcessor:
    """
    YOLO detection sonrası helmet/no-helmet sınıflandırmasını düzelten pipeline.
    """
    def __init__(self, ensemble_classifier, confidence_threshold=0.5, 
                 yolo_confidence_threshold=0.8):
        """
        Args:
            ensemble_classifier: EnsembleClassifier instance
            confidence_threshold: Post-classifier için minimum confidence
            yolo_confidence_threshold: Bu değerin üstündeki YOLO tespitleri direkt kabul edilir
        """
        self.ensemble = ensemble_classifier
        self.confidence_threshold = confidence_threshold
        self.yolo_confidence_threshold = yolo_confidence_threshold
        
        self.stats = {
            'total_detections': 0,
            'yolo_accepted': 0,
            'post_processed': 0,
            'corrected': 0,
            'rejected': 0
        }
    
    def process_detection(self, image, bbox, yolo_class, yolo_confidence):
        """
        Tek bir YOLO detection'ı işler.
        
        Args:
            image: Orijinal görüntü (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
            yolo_class: YOLO'nun tahmin ettiği class (0: no_helmet, 1: helmet)
            yolo_confidence: YOLO confidence score
        
        Returns:
            dict: {
                'final_class': int,
                'final_confidence': float,
                'corrected': bool,
                'method': str ('yolo_direct', 'post_processed', 'rejected')
            }
        """
        self.stats['total_detections'] += 1
        
        if yolo_confidence >= self.yolo_confidence_threshold:
            self.stats['yolo_accepted'] += 1
            return {
                'final_class': yolo_class,
                'final_confidence': yolo_confidence,
                'corrected': False,
                'method': 'yolo_direct',
                'bbox': bbox
            }
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            self.stats['rejected'] += 1
            return {
                'final_class': yolo_class,
                'final_confidence': 0.0,
                'corrected': False,
                'method': 'rejected',
                'bbox': bbox,
                'reason': 'invalid_bbox'
            }
        
        cropped, _ = crop_with_context(
            image,
            (x1, y1, x2, y2),
            **DEFAULT_CONTEXT_CROP_CONFIG,
        )
        
        if cropped.size == 0:
            self.stats['rejected'] += 1
            return {
                'final_class': yolo_class,
                'final_confidence': 0.0,
                'corrected': False,
                'method': 'rejected',
                'bbox': bbox,
                'reason': 'empty_crop'
            }
        
        temp_path = Path('/tmp/temp_crop.png')
        cv2.imwrite(str(temp_path), cropped)
        
        result = self.ensemble.predict(temp_path, return_details=True)
        
        self.stats['post_processed'] += 1
        
        if result['confidence'] < self.confidence_threshold:
            self.stats['rejected'] += 1
            return {
                'final_class': yolo_class,
                'final_confidence': yolo_confidence,
                'corrected': False,
                'method': 'low_confidence',
                'bbox': bbox,
                'ensemble_result': result
            }
        
        corrected = (result['prediction'] != yolo_class)
        if corrected:
            self.stats['corrected'] += 1
        
        return {
            'final_class': result['prediction'],
            'final_confidence': result['confidence'],
            'corrected': corrected,
            'method': 'post_processed',
            'bbox': bbox,
            'yolo_class': yolo_class,
            'yolo_confidence': yolo_confidence,
            'ensemble_result': result
        }
    
    def process_frame(self, image, detections):
        """
        Bir frame'deki tüm detection'ları işler.
        
        Args:
            image: Frame (numpy array)
            detections: List of dicts with keys: 'bbox', 'class', 'confidence'
        
        Returns:
            List of processed detections
        """
        results = []
        
        for det in detections:
            result = self.process_detection(
                image=image,
                bbox=det['bbox'],
                yolo_class=det['class'],
                yolo_confidence=det['confidence']
            )
            results.append(result)
        
        return results
    
    def visualize_results(self, image, results, output_path=None):
        """
        Sonuçları görselleştirir.
        
        Args:
            image: Orijinal görüntü
            results: process_frame() sonuçları
            output_path: Çıktı dosya yolu (None ise gösterir)
        """
        vis_image = image.copy()
        
        class_names = {0: 'no_helmet', 1: 'helmet'}
        colors = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            'corrected': (255, 165, 0)
        }
        
        for result in results:
            if result['method'] == 'rejected':
                continue
            
            x1, y1, x2, y2 = result['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            final_class = result['final_class']
            confidence = result['final_confidence']
            
            if result['corrected']:
                color = colors['corrected']
                label = f"{class_names[final_class]} {confidence:.2f} [CORRECTED]"
            else:
                color = colors[final_class]
                label = f"{class_names[final_class]} {confidence:.2f}"
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Görselleştirme kaydedildi: {output_path}")
        else:
            cv2.imshow('Results', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image
    
    def get_statistics(self):
        """
        İşlem istatistiklerini döner.
        """
        return {
            'total_detections': self.stats['total_detections'],
            'yolo_accepted': self.stats['yolo_accepted'],
            'post_processed': self.stats['post_processed'],
            'corrected': self.stats['corrected'],
            'rejected': self.stats['rejected'],
            'correction_rate': self.stats['corrected'] / self.stats['post_processed'] 
                              if self.stats['post_processed'] > 0 else 0
        }
    
    def print_statistics(self):
        """
        İstatistikleri yazdırır.
        """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 YOLO POST-PROCESSING İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam Detection: {stats['total_detections']}")
        print(f"YOLO Direkt Kabul: {stats['yolo_accepted']} "
              f"({100*stats['yolo_accepted']/stats['total_detections']:.1f}%)")
        print(f"Post-Processing: {stats['post_processed']} "
              f"({100*stats['post_processed']/stats['total_detections']:.1f}%)")
        print(f"Düzeltilen: {stats['corrected']} "
              f"({100*stats['correction_rate']:.1f}% of post-processed)")
        print(f"Reddedilen: {stats['rejected']}")
        print("="*60)
    
    def save_statistics(self, filepath='yolo_pipeline_stats.json'):
        """
        İstatistikleri JSON olarak kaydeder.
        """
        stats = self.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"✓ İstatistikler kaydedildi: {filepath}")

def demo_example():
    """
    Demo kullanım örneği.
    """
    print("="*60)
    print("YOLO POST-PROCESSING PIPELINE DEMO")
    print("="*60)
    
    ensemble = EnsembleClassifier(
        feature_model_path='feature_classifier_xgboost.pkl',
        cnn_model_path='cnn_classifier_best.pth',
        feature_weight=0.4,
        cnn_weight=0.6
    )
    
    pipeline = YOLOPostProcessor(
        ensemble_classifier=ensemble,
        confidence_threshold=0.7,
        yolo_confidence_threshold=0.8
    )
    
    print("\n✅ Pipeline hazır!")
    print("\nKullanım örneği:")
    print("""
    # YOLO detection sonuçları
    detections = [
        {'bbox': [100, 100, 200, 300], 'class': 1, 'confidence': 0.75},
        {'bbox': [300, 150, 400, 350], 'class': 0, 'confidence': 0.65},
    ]
    
    # Frame'i işle
    image = cv2.imread('frame.jpg')
    results = pipeline.process_frame(image, detections)
    
    # Görselleştir
    pipeline.visualize_results(image, results, 'output.jpg')
    
    # İstatistikleri göster
    pipeline.print_statistics()
    """)
    
    return pipeline

if __name__ == "__main__":
    demo_example()
