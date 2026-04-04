#!/usr/bin/env python3
"""
Streamlit Pipeline Dashboard — Helmet Detection Pipeline Visualization

Adımlar:
  1. Orijinal görüntü
  2. YOLO tespit kutucukları (sınıf etiketsiz)
  3. Kırpılmış tespitler (crop'lar)
  4. CNN sınıflandırma sonuçları
  5. Final annotated görüntü
"""

import os
import sys
import cv2
import torch
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV9_DIR = PROJECT_ROOT / "yolov9"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "e200_scratch.pt"
CNN_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "helmet_classifier_v5.pth"
INPUT_DIR = PROJECT_ROOT / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "300326_results"

# ── PyTorch 2.6+ compat ─────────────────────────────────────────────────────
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
from src.utils.image_ops import DEFAULT_CONTEXT_CROP_CONFIG, crop_with_context


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_yolo_model():
    device = "0" if torch.cuda.is_available() else "cpu"
    dev = select_device(device)
    model = DetectMultiBackend(str(YOLO_WEIGHTS), device=dev)
    stride = int(model.stride)
    imgsz = check_img_size(1280, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    return model, dev, imgsz, stride


@st.cache_resource
def get_cnn_model():
    classifier = CNNClassifier()
    classifier.load(str(CNN_WEIGHTS))
    classifier.model.eval()
    return classifier


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def yolo_detect(model, img_bgr, device, imgsz, stride, conf_thres, iou_thres):
    img_lb = letterbox(img_bgr, imgsz, stride=stride, auto=True)[0]
    img_t = img_lb.transpose((2, 0, 1))[::-1].copy()
    img_t = torch.from_numpy(img_t).to(device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    pred = model(img_t)
    pred = pred[0][1]
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=100)

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_t.shape[2:], det[:, :4], img_bgr.shape).round()
            for *xyxy, conf, cls in det:
                detections.append({
                    "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                    "conf": float(conf), "yolo_cls": int(cls),
                })
    return detections


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


COLORS = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255)}


def draw_yolo_boxes(img_bgr, detections):
    """YOLO tespitlerini (sınıfsız) sarı kutucuk olarak çizer."""
    out = img_bgr.copy()
    for i, d in enumerate(detections):
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 255), 2)
        text = f'Det {i+1} ({d["conf"]:.0%})'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (d["x1"], d["y1"] - th - 6), (d["x1"] + tw + 4, d["y1"]), (0, 255, 255), -1)
        cv2.putText(out, text, (d["x1"] + 2, d["y1"] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_final(img_bgr, results):
    """CNN sonuçlarıyla final annotated görüntü."""
    out = img_bgr.copy()
    for r in results:
        color = COLORS.get(r["label"], (255, 255, 0))
        cv2.rectangle(out, (r["x1"], r["y1"]), (r["x2"], r["y2"]), color, 2)
        text = f'{r["label"]} {r["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (r["x1"], r["y1"] - th - 8), (r["x1"] + tw + 4, r["y1"]), color, -1)
        cv2.putText(out, text, (r["x1"] + 2, r["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Helmet Detection Pipeline", layout="wide")
    st.title("Helmet Detection Pipeline")
    st.markdown("**YOLO Tespit → Kırpma → CNN Sınıflandırma → Sonuç**")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Ayarlar")
    conf_thres = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    iou_thres = st.sidebar.slider("YOLO IoU Threshold", 0.1, 0.9, 0.45, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Bilgileri**")
    st.sidebar.text(f"YOLO: {YOLO_WEIGHTS.name}")
    st.sidebar.text(f"CNN:  {CNN_WEIGHTS.name}")

    # ── Load models ──────────────────────────────────────────────────────────
    with st.spinner("Modeller yükleniyor..."):
        yolo_model, dev, imgsz, stride = get_yolo_model()
        cnn_classifier = get_cnn_model()
    st.sidebar.success(f"Modeller hazır (CNN: {cnn_classifier.branch_type})")

    # ── Image selection ──────────────────────────────────────────────────────
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in img_extensions]
    )

    if not image_paths:
        st.error(f"Görüntü bulunamadı: {INPUT_DIR}")
        return

    selected_name = st.selectbox(
        "Görüntü Seçin",
        [p.name for p in image_paths],
        index=0,
    )
    selected_path = INPUT_DIR / selected_name

    # ── Run pipeline button ──────────────────────────────────────────────────
    if st.button("Pipeline Çalıştır", type="primary"):
        img_bgr = cv2.imread(str(selected_path))
        if img_bgr is None:
            st.error("Görüntü okunamadı!")
            return

        h, w = img_bgr.shape[:2]

        # ── STEP 1: Original ─────────────────────────────────────────────
        st.markdown("---")
        st.header("Adım 1: Orijinal Görüntü")
        st.image(bgr_to_rgb(img_bgr), caption=f"{selected_name} ({w}x{h})", width="stretch")

        # ── STEP 2: YOLO Detection ───────────────────────────────────────
        st.markdown("---")
        st.header("Adım 2: YOLO Tespitleri")
        with st.spinner("YOLO inference..."):
            detections = yolo_detect(yolo_model, img_bgr, dev, imgsz, stride, conf_thres, iou_thres)

        st.success(f"{len(detections)} tespit bulundu")
        yolo_vis = draw_yolo_boxes(img_bgr, detections)
        st.image(bgr_to_rgb(yolo_vis), caption="YOLO tespitleri (sınıf etiketsiz, sadece bbox)", width="stretch")

        if not detections:
            st.warning("Hiç tespit bulunamadı. Confidence threshold'u düşürmeyi deneyin.")
            return

        # ── STEP 3: Crop Detections ──────────────────────────────────────
        st.markdown("---")
        st.header("Adım 3: Kırpılmış Tespitler")

        crops = []
        valid_detections = []
        for d in detections:
            crop, _ = crop_with_context(
                img_bgr,
                (d["x1"], d["y1"], d["x2"], d["y2"]),
                **DEFAULT_CONTEXT_CROP_CONFIG,
            )
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_detections.append(d)

        cols = st.columns(min(len(crops), 6))
        for i, crop in enumerate(crops):
            with cols[i % len(cols)]:
                st.image(bgr_to_rgb(crop), caption=f"Det {i+1}", width="stretch")

        # ── STEP 4: CNN Classification (batch) ──────────────────────────
        st.markdown("---")
        st.header("Adım 4: CNN Sınıflandırma")

        # Batch inference
        crops_pil = [Image.fromarray(bgr_to_rgb(c)) for c in crops]
        batch_results = cnn_classifier.predict_batch(crops_pil, return_proba=True)
        labels_map = {0: "no_helmet", 1: "helmet"}

        results = []
        cols = st.columns(min(len(crops), 6))
        for i, (crop, det, br) in enumerate(zip(crops, valid_detections, batch_results)):
            label = labels_map[br["prediction"]]
            confidence = br["confidence"]
            proba_helmet = br["probabilities"]["helmet"]
            proba_no_helmet = br["probabilities"]["no_helmet"]
            result = {
                "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                "yolo_conf": det["conf"],
                "yolo_cls": det["yolo_cls"],
                "label": label,
                "confidence": confidence,
                "proba_helmet": proba_helmet,
                "proba_no_helmet": proba_no_helmet,
            }
            results.append(result)

            with cols[i % len(cols)]:
                emoji = "✅" if label == "helmet" else "❌"
                color = "green" if label == "helmet" else "red"
                st.image(bgr_to_rgb(crop), width="stretch")
                st.markdown(
                    f'{emoji} **:{color}[{label}]** '
                    f'({confidence:.0%})'
                )
                st.caption(
                    f'Helmet: {proba_helmet:.1%} | '
                    f'No helmet: {proba_no_helmet:.1%}'
                )

        # ── STEP 5: Final Result ─────────────────────────────────────────
        st.markdown("---")
        st.header("Adım 5: Sonuç")

        final_img = draw_final(img_bgr, results)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / selected_name
        cv2.imwrite(str(save_path), final_img)

        st.image(bgr_to_rgb(final_img), caption="Final sonuç", width="stretch")
        st.success(f"Kaydedildi: {save_path}")

        # ── Summary ──────────────────────────────────────────────────────
        st.markdown("---")
        st.header("Özet")
        n_helmet = sum(1 for r in results if r["label"] == "helmet")
        n_no = sum(1 for r in results if r["label"] == "no_helmet")

        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Tespit", len(results))
        col2.metric("Helmet", n_helmet)
        col3.metric("No Helmet", n_no)

        st.dataframe(
            [
                {
                    "Det": i + 1,
                    "Label": r["label"],
                    "CNN Confidence": f'{r["confidence"]:.1%}',
                    "YOLO Confidence": f'{r["yolo_conf"]:.1%}',
                    "YOLO Class": r["yolo_cls"],
                    "BBox": f'({r["x1"]},{r["y1"]})–({r["x2"]},{r["y2"]})',
                }
                for i, r in enumerate(results)
            ],
        )

    # ── Batch mode ───────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Tüm Görüntüleri İşle"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        progress = st.progress(0)
        status = st.empty()

        total_helmet = 0
        total_no = 0

        for idx, img_path in enumerate(image_paths):
            status.text(f"İşleniyor: {img_path.name} ({idx+1}/{len(image_paths)})")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            detections = yolo_detect(yolo_model, img_bgr, dev, imgsz, stride, conf_thres, iou_thres)

            # Batch CNN inference
            crops_pil = []
            valid_detections = []
            for d in detections:
                crop, _ = crop_with_context(
                    img_bgr,
                    (d["x1"], d["y1"], d["x2"], d["y2"]),
                    **DEFAULT_CONTEXT_CROP_CONFIG,
                )
                if crop.size == 0:
                    continue
                crops_pil.append(Image.fromarray(bgr_to_rgb(crop)))
                valid_detections.append(d)
            results = []
            if crops_pil:
                batch_res = cnn_classifier.predict_batch(crops_pil, return_proba=True)
                lmap = {0: "no_helmet", 1: "helmet"}
                for d, br in zip(valid_detections, batch_res):
                    results.append({
                        **d,
                        "label": lmap[br["prediction"]],
                        "confidence": br["confidence"],
                        "proba_helmet": br["probabilities"]["helmet"],
                        "proba_no_helmet": br["probabilities"]["no_helmet"],
                    })

            final_img = draw_final(img_bgr, results)
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), final_img)

            total_helmet += sum(1 for r in results if r["label"] == "helmet")
            total_no += sum(1 for r in results if r["label"] == "no_helmet")

            progress.progress((idx + 1) / len(image_paths))

        status.empty()
        progress.empty()
        st.success(
            f"Tamamlandı! {len(image_paths)} görüntü işlendi. "
            f"Helmet: {total_helmet}, No helmet: {total_no}. "
            f"Sonuçlar: {OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()
