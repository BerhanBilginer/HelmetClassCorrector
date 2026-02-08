# HelmetClassCorrector - Yeni Proje Yapısı

## 📁 Düzenlenmiş Dizin Yapısı

```
HelmetClassCorrector/
│
├── 📂 src/                          # Kaynak kod
│   ├── __init__.py
│   │
│   ├── 📂 data_preparation/         # Veri hazırlama scriptleri
│   │   ├── __init__.py
│   │   ├── label_manipulator.py     # YOLO label filtreleme
│   │   ├── cropper.py               # Obje cropping
│   │   └── prepare_dataset.py       # Train/val/test split
│   │
│   ├── 📂 models/                   # Model tanımları ve training
│   │   ├── __init__.py
│   │   ├── feature_classifier.py    # XGBoost classifier
│   │   ├── cnn_classifier.py        # CNN classifier
│   │   └── ensemble_classifier.py   # Ensemble system
│   │
│   ├── 📂 evaluation/               # Test ve değerlendirme
│   │   ├── __init__.py
│   │   ├── test_yolo_simulation.py  # YOLO simulation testi
│   │   ├── analyze_images.py        # Görüntü analizi
│   │   └── streamlit_app.py         # Streamlit dashboard
│   │
│   └── 📂 utils/                    # Yardımcı araçlar
│       ├── __init__.py
│       └── yolo_pipeline.py         # YOLO entegrasyon pipeline
│
├── 📂 models/                       # Eğitilmiş modeller
│   └── trained/
│       ├── feature_classifier_xgboost.pkl
│       ├── cnn_classifier_best.pth
│       └── .gitkeep
│
├── 📂 config/                       # Konfigürasyon dosyaları
│   ├── ensemble_config.json
│   ├── requirements_streamlit.txt
│   └── .gitkeep
│
├── 📂 results/                      # Sonuçlar ve çıktılar
│   ├── visualizations/              # Grafikler ve görseller
│   │   ├── *_confusion_matrix.png
│   │   ├── *_importance.png
│   │   ├── *_training_history.png
│   │   └── yolo_simulation_visualization.png
│   │
│   └── reports/                     # Raporlar ve metrikler
│       ├── yolo_simulation_results.json
│       ├── image_analysis.csv
│       └── image_analysis.xlsx
│
├── 📂 docs/                         # Dokümantasyon
│   ├── PROJECT_STRUCTURE.md
│   ├── IMPROVEMENT_ANALYSIS.md
│   └── NEW_PROJECT_STRUCTURE.md     # Bu dosya
│
├── 📂 test/                         # Test verileri
│   ├── cropped_images/              # Orijinal crop'lar
│   │   ├── helmet/
│   │   └── no_helmet/
│   │
│   ├── cnn_test/                    # YOLO simulation test
│   │   ├── helmet/
│   │   └── no_helmet/
│   │
│   └── images/                      # Test görüntüleri
│
├── 📂 dataset/                      # Hazırlanmış dataset
│   ├── train/
│   │   ├── helmet/
│   │   └── no_helmet/
│   ├── val/
│   │   ├── helmet/
│   │   └── no_helmet/
│   └── test/
│       ├── helmet/
│       └── no_helmet/
│
├── 📂 kiran_dataset/                # YOLO format dataset
│   ├── train/
│   ├── val/
│   └── test/
│
├── 📄 train_all.py                  # Master training script
├── 📄 run_test.py                   # Quick test script
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Git ignore rules
└── 📄 README.md                     # Ana dokümantasyon
```

---

## 🎯 Klasör Açıklamaları

### `src/` - Kaynak Kod
Tüm Python scriptleri modüler bir yapıda organize edildi.

**`data_preparation/`** - Veri hazırlama
- YOLO label'ları filtreleme
- Objeleri cropping
- Dataset'i train/val/test'e bölme

**`models/`** - Model tanımları
- Feature-based classifier (XGBoost)
- CNN classifier (PyTorch)
- Ensemble system

**`evaluation/`** - Test ve değerlendirme
- YOLO simulation testi
- Görüntü analizi
- Streamlit dashboard

**`utils/`** - Yardımcı araçlar
- YOLO entegrasyon pipeline
- Ortak fonksiyonlar

### `models/trained/` - Eğitilmiş Modeller
Tüm trained model dosyaları burada saklanır.

### `config/` - Konfigürasyon
Model konfigürasyonları ve ayar dosyaları.

### `results/` - Sonuçlar
**`visualizations/`** - Tüm grafikler ve görseller
**`reports/`** - JSON, CSV, XLSX raporları

### `docs/` - Dokümantasyon
Proje dokümantasyonu ve analizler.

---

## 🚀 Kullanım

### Veri Hazırlama
```bash
# 1. YOLO label'ları filtrele
python -m src.data_preparation.label_manipulator

# 2. Objeleri crop et
python -m src.data_preparation.cropper

# 3. Dataset'i böl
python -m src.data_preparation.prepare_dataset
```

### Model Eğitimi
```bash
# Tüm modelleri eğit (önerilen)
python train_all.py

# Veya tek tek:
python -m src.models.feature_classifier
python -m src.models.cnn_classifier
python -m src.models.ensemble_classifier
```

### Test ve Değerlendirme
```bash
# Hızlı test
python run_test.py

# YOLO simulation
python -m src.evaluation.test_yolo_simulation

# Streamlit dashboard
streamlit run src/evaluation/streamlit_app.py
```

### YOLO Entegrasyonu
```python
import sys
sys.path.insert(0, '.')

from src.models.ensemble_classifier import EnsembleClassifier
from src.utils.yolo_pipeline import YOLOPostProcessor

# Modelleri yükle
ensemble = EnsembleClassifier(
    feature_model_path='models/trained/feature_classifier_xgboost.pkl',
    cnn_model_path='models/trained/cnn_classifier_best.pth'
)

# Pipeline oluştur
pipeline = YOLOPostProcessor(ensemble_classifier=ensemble)

# Kullan
results = pipeline.process_frame(image, yolo_detections)
```

---

## 📦 Import Path'leri

### Eski Yapı
```python
from feature_classifier import FeatureBasedClassifier
from cnn_classifier import CNNClassifier
```

### Yeni Yapı
```python
from src.models.feature_classifier import FeatureBasedClassifier
from src.models.cnn_classifier import CNNClassifier
```

---

## ✅ Avantajlar

1. **Modüler Yapı:** Her şey mantıklı klasörlerde
2. **Kolay Navigasyon:** Dosyaları bulmak çok kolay
3. **Temiz Root:** Ana dizin karmaşık değil
4. **Scalable:** Yeni modüller eklemek kolay
5. **Professional:** Standart Python proje yapısı
6. **Git-friendly:** .gitignore ile düzenli

---

## 🔄 Migrasyon Notları

Eski scriptler otomatik olarak taşındı. Eğer eski import path'leri kullanan custom scriptleriniz varsa, bunları güncellemeniz gerekir:

```python
# Eski
from cropper import crop_objects_from_dataset

# Yeni
from src.data_preparation.cropper import crop_objects_from_dataset
```

---

## 📝 Sonraki Adımlar

1. ✅ Proje reorganize edildi
2. ✅ Import path'leri güncellendi
3. ✅ Dokümantasyon oluşturuldu
4. 🔄 Tüm scriptleri test et
5. 🔄 README'yi güncelle
6. 🔄 Git commit yap

---

**Düzenli bir proje = Mutlu bir developer! 🎉**
