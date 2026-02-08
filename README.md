# HelmetClassCorrector 🪖

**Post-processing classifier for YOLO helmet detection - Corrects false positives and false negatives**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

YOLOv9 helmet detection model sometimes misclassifies:
- Helmet-wearing persons as non-helmet
- Non-helmet persons as helmet-wearing

**Solution:** A two-stage hybrid post-processing system that corrects YOLO predictions.

---

## 🏗️ System Architecture

```
YOLO Detection → Crop Object → Post-Classifier → Corrected Prediction
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Feature-Based (40%)              CNN (60%)
            (XGBoost)                    (Lightweight CNN)
            - Brightness                 - Visual patterns
            - Sharpness                  - Texture features
            - Saturation                 - Edge detection
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                            Weighted Ensemble
                                    ↓
                          Final Prediction
```

---

## 📊 Performance

| Model | Accuracy | Precision | Recall | Speed |
|-------|----------|-----------|--------|-------|
| **Feature-based (XGBoost)** | 74.72% | 0.80 | 0.54 | <1ms |
| **CNN Classifier** | **97.06%** | 0.97 | 0.96 | ~5ms |
| **Ensemble** | ~95-97% | ~0.97 | ~0.96 | ~6ms |

### YOLO Simulation Test
- **Test images:** 10 (5 helmet + 5 no_helmet)
- **Corrections made:** 5/10 (50%)
- **Agreement with YOLO:** 5/10 (50%)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd HelmetClassCorrector

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Step 1: Filter YOLO labels (keep only class 1 & 3)
python label_manipulator.py

# Step 2: Crop objects from YOLO dataset
python cropper.py

# Step 3: Split into train/val/test
python prepare_dataset.py
```

### 3. Train Models

```bash
# Train all models at once (recommended)
python train_all.py

# Or train individually:
python feature_classifier.py  # XGBoost
python cnn_classifier.py      # CNN
python ensemble_classifier.py # Ensemble
```

### 4. Test & Evaluate

```bash
# Run YOLO simulation test
python test_yolo_simulation.py

# Visualize results
streamlit run streamlit_app.py
```

---

## 💻 Usage

### Basic Usage

```python
from ensemble_classifier import EnsembleClassifier

# Load models
ensemble = EnsembleClassifier(
    feature_model_path='feature_classifier_xgboost.pkl',
    cnn_model_path='cnn_classifier_best.pth',
    feature_weight=0.4,
    cnn_weight=0.6
)

# Predict single image
result = ensemble.predict('image.jpg', return_details=True)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### YOLO Integration

```python
from yolo_pipeline import YOLOPostProcessor
from ensemble_classifier import EnsembleClassifier
import cv2

# Initialize
ensemble = EnsembleClassifier(
    feature_model_path='feature_classifier_xgboost.pkl',
    cnn_model_path='cnn_classifier_best.pth'
)

pipeline = YOLOPostProcessor(
    ensemble_classifier=ensemble,
    confidence_threshold=0.7,
    yolo_confidence_threshold=0.8
)

# Process frame
image = cv2.imread('frame.jpg')

# YOLO detections format
yolo_detections = [
    {'bbox': [x1, y1, x2, y2], 'class': 1, 'confidence': 0.75},
    {'bbox': [x1, y1, x2, y2], 'class': 0, 'confidence': 0.65},
]

# Post-process
results = pipeline.process_frame(image, yolo_detections)

# Visualize
pipeline.visualize_results(image, results, 'output.jpg')
pipeline.print_statistics()
```

---

## 📁 Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory layout.

```
HelmetClassCorrector/
├── kiran_dataset/          # YOLO format dataset
├── test/cropped_images/    # Cropped training images
├── dataset/                # Prepared train/val/test split
├── *.py                    # Python scripts
├── *.pkl, *.pth           # Trained models
└── *.png, *.json          # Results & visualizations
```

---

## 🔧 Configuration

### Ensemble Weights

Adjust in `ensemble_classifier.py`:
```python
ensemble = EnsembleClassifier(
    feature_weight=0.4,  # Feature-based model weight
    cnn_weight=0.6       # CNN model weight
)
```

### Confidence Thresholds

Adjust in `yolo_pipeline.py`:
```python
pipeline = YOLOPostProcessor(
    confidence_threshold=0.7,           # Min confidence for post-classifier
    yolo_confidence_threshold=0.8       # Direct accept YOLO if above this
)
```

---

## 📈 Training Details

### Dataset
- **Total images:** 37,570
- **Train:** 26,298 (70%)
- **Val:** 5,635 (15%)
- **Test:** 5,637 (15%)
- **Classes:** Helmet (42.5%), No-Helmet (57.5%)

### Feature-Based Classifier
- **Algorithm:** XGBoost
- **Features:** Brightness, Sharpness, Saturation
- **Training time:** ~2 minutes

### CNN Classifier
- **Architecture:** 3 Conv + 2 Dense layers
- **Input:** 64x64 RGB
- **Epochs:** 20
- **Training time:** ~20 minutes (CPU)
- **Data augmentation:** Rotation, flip, color jitter

---

## 🎨 Visualizations

The project generates several visualizations:
- Confusion matrices
- Feature importance plots
- Training history curves
- YOLO simulation results

---

## 🐛 Known Issues

1. **CUDA initialization error:** Models currently train on CPU
   - Workaround: CPU training works fine, just slower
   - Fix: Restart Python environment before training

2. **Font warnings in matplotlib:** Unicode characters not rendering
   - Impact: Minor visual issue in plots
   - Workaround: Ignore warnings

---

## 🔮 Future Improvements

1. **GPU Support:** Fix CUDA initialization for faster training
2. **Transfer Learning:** Try MobileNetV3 or EfficientNet
3. **Hyperparameter Tuning:** Grid search for optimal weights
4. **Data Augmentation:** More sophisticated augmentation
5. **Real-world Testing:** Deploy on actual YOLO outputs
6. **Active Learning:** Collect hard examples and retrain
7. **Model Compression:** Quantization for faster inference
8. **API Service:** REST API for production deployment

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{helmetclasscorrector2026,
  title={HelmetClassCorrector: Post-processing for YOLO Helmet Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/HelmetClassCorrector}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

**Made with ❤️ for safer workplaces**