#!/bin/bash

# Data preparation scripts
mv label_manipulator.py src/data_preparation/
mv cropper.py src/data_preparation/
mv prepare_dataset.py src/data_preparation/

# Model scripts
mv feature_classifier.py src/models/
mv cnn_classifier.py src/models/
mv ensemble_classifier.py src/models/

# Evaluation scripts
mv test_yolo_simulation.py src/evaluation/
mv analyze_images.py src/evaluation/
mv streamlit_app.py src/evaluation/

# Pipeline/Utils
mv yolo_pipeline.py src/utils/

# Training script (keep in root for easy access)
# train_all.py stays in root

# Trained models
mv feature_classifier_xgboost.pkl models/trained/ 2>/dev/null || true
mv cnn_classifier_best.pth models/trained/ 2>/dev/null || true
mv ensemble_config.json config/ 2>/dev/null || true

# Visualizations
mv *_confusion_matrix.png results/visualizations/ 2>/dev/null || true
mv *_importance.png results/visualizations/ 2>/dev/null || true
mv *_training_history.png results/visualizations/ 2>/dev/null || true
mv yolo_simulation_visualization.png results/visualizations/ 2>/dev/null || true

# Reports/Results
mv yolo_simulation_results.json results/reports/ 2>/dev/null || true
mv image_analysis.csv results/reports/ 2>/dev/null || true
mv image_analysis.xlsx results/reports/ 2>/dev/null || true

# Documentation
mv PROJECT_STRUCTURE.md docs/
mv IMPROVEMENT_ANALYSIS.md docs/

# Config files
mv requirements_streamlit.txt config/

echo "✅ Reorganization complete!"
