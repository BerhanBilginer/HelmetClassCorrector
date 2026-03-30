#!/usr/bin/env python3
"""
Pipeline Test: YOLO Detection → Crop → CNN Classification → Annotate → Save

YOLO modeli kafa bölgelerini tespit eder (sınıf etiketi kullanılmaz),
CNN (HelmetClassifierNet v5) her kırpılmış bölgeyi helmet/no_helmet olarak sınıflandırır,
sonuç orijinal görüntü üzerine çizilir ve kaydedilir.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV9_DIR = PROJECT_ROOT / "yolov9"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "e200_scratch.pt"
CNN_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "helmet_classifier_v4.pth"
INPUT_DIR = PROJECT_ROOT / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "300326_results"

# ── PyTorch 2.6+ compat for YOLOv9 pickle loading ───────────────────────────
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ── YOLOv9 imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(YOLOV9_DIR))
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ── CNN import ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.cnn_classifier import CNNClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# 1. YOLO MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolo_model(weights=YOLO_WEIGHTS, device="0", imgsz=1280):
    """YOLOv9 modelini yükler."""
    dev = select_device(device)
    model = DetectMultiBackend(str(weights), device=dev)
    stride = int(model.stride)
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    print(f"✓ YOLO model yüklendi: {weights.name}")
    print(f"  Sınıflar: {model.names}")
    return model, dev, imgsz, stride


# ═══════════════════════════════════════════════════════════════════════════════
# 2. YOLO INFERENCE — sadece bbox koordinatları
# ═══════════════════════════════════════════════════════════════════════════════

def yolo_detect(model, img_bgr, device, imgsz, stride, conf_thres=0.25, iou_thres=0.45):
    """
    YOLO inference yapar ve tespit kutucuklarını döndürür.
    Returns: list of dict  [{x1, y1, x2, y2, conf, yolo_cls}, ...]
    """
    img_letterboxed = letterbox(img_bgr, imgsz, stride=stride, auto=True)[0]
    img_t = img_letterboxed.transpose((2, 0, 1))[::-1].copy()  # BGR→RGB, HWC→CHW
    img_t = torch.from_numpy(img_t).to(device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    pred = model(img_t)
    pred = pred[0][1]  # dual head
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=100)

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_t.shape[2:], det[:, :4], img_bgr.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(conf),
                    "yolo_cls": int(cls),
                })
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CNN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_cnn_model(weights=CNN_WEIGHTS):
    """CNN modelini yükler (v4/v5 uyumlu)."""
    classifier = CNNClassifier()
    classifier.load(str(weights))
    classifier.model.eval()
    print(f"✓ CNN model yüklendi: {weights.name} (branch: {classifier.branch_type})")
    return classifier


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANNOTATE & SAVE
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255)}


def annotate_image(img_bgr, results):
    """
    Tespit sonuçlarını görüntü üzerine çizer.
    results: list of dict [{x1,y1,x2,y2, label, confidence}, ...]
    """
    annotated = img_bgr.copy()
    for r in results:
        color = COLORS.get(r["label"], (255, 255, 0))
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f'{r["label"]} {r["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                 yolo_weights=YOLO_WEIGHTS, cnn_weights=CNN_WEIGHTS,
                 conf_thres=0.25, iou_thres=0.45, device="0"):
    """
    Tüm pipeline'ı çalıştırır.
    Returns: list of per-image result dicts (Streamlit tarafından da kullanılır)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HELMET DETECTION PIPELINE")
    print("=" * 60)

    # Load models
    yolo_model, dev, imgsz, stride = load_yolo_model(yolo_weights, device)
    cnn_classifier = load_cnn_model(cnn_weights)

    # Find images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        [p for p in Path(input_dir).iterdir() if p.suffix.lower() in img_extensions]
    )
    print(f"\n📁 {len(image_paths)} görüntü bulundu: {input_dir}")
    print("=" * 60)

    all_results = []

    for img_path in image_paths:
        print(f"\n📸 İşleniyor: {img_path.name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ⚠️  Okunamadı, atlanıyor.")
            continue

        # YOLO detect
        detections = yolo_detect(yolo_model, img_bgr, dev, imgsz, stride, conf_thres, iou_thres)
        print(f"  🔍 YOLO: {len(detections)} tespit")

        # Collect crops for batch inference
        crops_pil = []
        for det in detections:
            crop_bgr = img_bgr[det["y1"]:det["y2"], det["x1"]:det["x2"]]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crops_pil.append(Image.fromarray(crop_rgb))

        # Batch CNN classification
        image_results = []
        if crops_pil:
            batch_results = cnn_classifier.predict_batch(crops_pil, return_proba=True)
            labels_map = {0: "no_helmet", 1: "helmet"}

            for i, (det, br) in enumerate(zip(detections, batch_results)):
                label = labels_map[br["prediction"]]
                result = {
                    "x1": det["x1"], "y1": det["y1"],
                    "x2": det["x2"], "y2": det["y2"],
                    "yolo_conf": det["conf"],
                    "yolo_cls": det["yolo_cls"],
                    "label": label,
                    "confidence": br["confidence"],
                    "proba_helmet": br["probabilities"]["helmet"],
                    "proba_no_helmet": br["probabilities"]["no_helmet"],
                }
                image_results.append(result)

                tag = "✅" if label == "helmet" else "❌"
                print(f"  {tag} Det {i+1}: {label} ({br['confidence']:.1%})")

        # Annotate & save
        annotated = annotate_image(img_bgr, image_results)
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)
        print(f"  💾 Kaydedildi: {save_path.name}")

        all_results.append({
            "image_path": str(img_path),
            "image_name": img_path.name,
            "save_path": str(save_path),
            "detections": image_results,
            "num_helmet": sum(1 for r in image_results if r["label"] == "helmet"),
            "num_no_helmet": sum(1 for r in image_results if r["label"] == "no_helmet"),
        })

    # Summary
    total_det = sum(len(r["detections"]) for r in all_results)
    total_helmet = sum(r["num_helmet"] for r in all_results)
    total_no_helmet = sum(r["num_no_helmet"] for r in all_results)

    print("\n" + "=" * 60)
    print("✅ PIPELINE TAMAMLANDI")
    print("=" * 60)
    print(f"  Görüntü: {len(all_results)}")
    print(f"  Toplam tespit: {total_det}")
    print(f"  Helmet: {total_helmet}  |  No helmet: {total_no_helmet}")
    print(f"  Sonuçlar: {output_dir}")
    print("=" * 60)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_pipeline()