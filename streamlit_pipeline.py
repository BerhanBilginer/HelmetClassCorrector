#!/usr/bin/env python3
"""
Streamlit Pipeline Dashboard — Helmet Detection Pipeline Visualization

Adımlar:
  1. Orijinal görüntü
  2. YOLO tespit kutucukları (sınıf etiketsiz)
  3. Kırpılmış tespitler (crop'lar)
  4. CNN sınıflandırma sonuçları
  5. Final annotated görüntü
  6. Grad-CAM aktivasyon analizi
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV9_DIR = PROJECT_ROOT / "yolov9"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "e200_scratch.pt"
CNN_WEIGHTS = PROJECT_ROOT / "models" / "trained" / "helmet_classifier_center_guided_v1.pth"
INPUT_DIR = PROJECT_ROOT / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "060426_results"
ACTIVATION_OUTPUT_DIR = PROJECT_ROOT / "results" / "activation_debug"
ACTIVATION_COMPARISON_OUTPUT_DIR = PROJECT_ROOT / "results" / "activation_debug_dual_target"

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
    return load_yolo_model()


def load_yolo_model():
    device = "0" if torch.cuda.is_available() else "cpu"
    dev = select_device(device)
    model = DetectMultiBackend(str(YOLO_WEIGHTS), device=dev)
    stride = int(model.stride)
    imgsz = check_img_size(1280, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    return model, dev, imgsz, stride


@st.cache_resource
def get_cnn_model():
    return load_cnn_model()


def load_cnn_model(weights_path=None):
    classifier = CNNClassifier()
    model_path = Path(weights_path) if weights_path is not None else CNN_WEIGHTS
    classifier.load(str(model_path))
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


def slugify_filename(text):
    """Metni dosya adı için güvenli hale getir."""
    safe = [
        ch.lower() if ch.isalnum() else "_"
        for ch in text
    ]
    return "".join(safe).strip("_")


def draw_selected_bbox(image_rgb, result, color=(255, 165, 0)):
    """Seçilen detection bbox'unu tam görüntü üzerinde vurgula."""
    highlighted = image_rgb.copy()
    x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
    cv2.rectangle(highlighted, (x1, y1), (x2, y2), color, 3)
    label = f"{result['label']} {result['confidence']:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = max(0, y1 - th - 10)
    cv2.rectangle(highlighted, (x1, top), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        highlighted,
        label,
        (x1 + 3, max(th + 2, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return highlighted


def resolve_target_mode(target_mode, predicted_label):
    """UI/CLI hedef bilgisini class id + label formatına çevir."""
    if target_mode == "prediction":
        return None, predicted_label
    return (1 if target_mode == "helmet" else 0), target_mode


def get_target_gradcam_views(cnn_classifier, crop_pil, target_mode, layers=None, predicted_label=None):
    """Seçilen hedef için tüm katmanlardan Grad-CAM sonuçlarını üret."""
    if layers is None:
        layers = get_gradcam_layers(cnn_classifier)

    target_class, target_label = resolve_target_mode(
        target_mode,
        predicted_label or "prediction",
    )
    gradcam_views = {
        name: cnn_classifier.gradcam(crop_pil, target_class=target_class, target_layer=layer)
        for name, layer in layers.items()
    }
    return target_label, gradcam_views


def save_gradcam_layers(bundle_dir, gradcam_views):
    """Katman bazlı overlay/heatmap çıktısını belirtilen klasöre kaydeder."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    layer_manifest = {}
    for layer_name, cam_result in gradcam_views.items():
        layer_slug = slugify_filename(layer_name)
        overlay_path = bundle_dir / f"{layer_slug}_overlay.png"
        heatmap_png_path = bundle_dir / f"{layer_slug}_heatmap.png"
        heatmap_npy_path = bundle_dir / f"{layer_slug}_heatmap.npy"

        Image.fromarray(cam_result["overlay"]).save(overlay_path)
        heatmap_uint8 = np.clip(cam_result["heatmap"] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(heatmap_uint8).save(heatmap_png_path)
        np.save(heatmap_npy_path, cam_result["heatmap"])

        layer_manifest[layer_name] = {
            "overlay_file": overlay_path.name,
            "heatmap_png_file": heatmap_png_path.name,
            "heatmap_npy_file": heatmap_npy_path.name,
            "prediction": int(cam_result["prediction"]),
            "probabilities": {
                "helmet": float(cam_result["probabilities"]["helmet"]),
                "no_helmet": float(cam_result["probabilities"]["no_helmet"]),
            },
        }

    return layer_manifest


def save_activation_bundle(
    pipeline_data,
    detection_idx,
    target_label,
    gradcam_views,
    output_root=ACTIVATION_OUTPUT_DIR,
):
    """
    Seçilen crop için aktivasyon analizini diske kaydeder.

    Kaydedilenler:
      - orijinal görüntü
      - bbox vurgulu görüntü
      - YOLO tespit görüntüsü
      - final annotate görüntüsü
      - seçili crop
      - her katman için overlay png + heatmap png + heatmap npy
      - metadata.json
    """
    result = pipeline_data["results"][detection_idx]
    image_stem = Path(pipeline_data["image_name"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_slug = slugify_filename(target_label)
    bundle_dir = (
        output_root
        / image_stem
        / f"det_{detection_idx + 1:02d}_{result['label']}_target_{target_slug}_{timestamp}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(pipeline_data["image_rgb"]).save(bundle_dir / "original_image.png")
    Image.fromarray(draw_selected_bbox(pipeline_data["image_rgb"], result)).save(
        bundle_dir / "selected_bbox.png"
    )
    Image.fromarray(pipeline_data["yolo_vis_rgb"]).save(bundle_dir / "yolo_detections.png")
    Image.fromarray(pipeline_data["final_rgb"]).save(bundle_dir / "final_annotated.png")
    Image.fromarray(pipeline_data["crops_rgb"][detection_idx]).save(bundle_dir / "selected_crop.png")

    layer_manifest = save_gradcam_layers(bundle_dir, gradcam_views)

    metadata = {
        "image_name": pipeline_data["image_name"],
        "image_path": pipeline_data["image_path"],
        "pipeline_output_path": pipeline_data["save_path"],
        "saved_at": timestamp,
        "detection_index": detection_idx,
        "target_label": target_label,
        "bbox": {
            "x1": int(result["x1"]),
            "y1": int(result["y1"]),
            "x2": int(result["x2"]),
            "y2": int(result["y2"]),
        },
        "prediction": {
            "label": result["label"],
            "class_id": int(result["prediction"]),
            "confidence": float(result["confidence"]),
            "helmet_probability": float(result["proba_helmet"]),
            "no_helmet_probability": float(result["proba_no_helmet"]),
        },
        "yolo": {
            "confidence": float(result["yolo_conf"]),
            "class_id": int(result["yolo_cls"]),
        },
        "context_crop_config": DEFAULT_CONTEXT_CROP_CONFIG,
        "layers": layer_manifest,
    }

    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return bundle_dir


def save_activation_comparison_bundle(
    pipeline_data,
    detection_idx,
    target_views,
    output_root=ACTIVATION_COMPARISON_OUTPUT_DIR,
):
    """
    Aynı crop için helmet ve no_helmet hedeflerini birlikte kaydeder.

    Yapı:
      bundle/
        original_image.png
        selected_bbox.png
        yolo_detections.png
        final_annotated.png
        selected_crop.png
        helmet/
        no_helmet/
        metadata.json
    """
    result = pipeline_data["results"][detection_idx]
    image_stem = Path(pipeline_data["image_name"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = (
        output_root
        / image_stem
        / f"det_{detection_idx + 1:02d}_pred_{result['label']}_{timestamp}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(pipeline_data["image_rgb"]).save(bundle_dir / "original_image.png")
    Image.fromarray(draw_selected_bbox(pipeline_data["image_rgb"], result)).save(
        bundle_dir / "selected_bbox.png"
    )
    Image.fromarray(pipeline_data["yolo_vis_rgb"]).save(bundle_dir / "yolo_detections.png")
    Image.fromarray(pipeline_data["final_rgb"]).save(bundle_dir / "final_annotated.png")
    Image.fromarray(pipeline_data["crops_rgb"][detection_idx]).save(bundle_dir / "selected_crop.png")

    targets_manifest = {}
    for target_label, gradcam_views in target_views.items():
        target_dir = bundle_dir / slugify_filename(target_label)
        layer_manifest = save_gradcam_layers(target_dir, gradcam_views)
        targets_manifest[target_label] = {
            "class_id": 1 if target_label == "helmet" else 0,
            "layers": layer_manifest,
        }

    metadata = {
        "image_name": pipeline_data["image_name"],
        "image_path": pipeline_data["image_path"],
        "pipeline_output_path": pipeline_data["save_path"],
        "saved_at": timestamp,
        "bundle_type": "dual_target_comparison",
        "detection_index": detection_idx,
        "bbox": {
            "x1": int(result["x1"]),
            "y1": int(result["y1"]),
            "x2": int(result["x2"]),
            "y2": int(result["y2"]),
        },
        "prediction": {
            "label": result["label"],
            "class_id": int(result["prediction"]),
            "confidence": float(result["confidence"]),
            "helmet_probability": float(result["proba_helmet"]),
            "no_helmet_probability": float(result["proba_no_helmet"]),
        },
        "yolo": {
            "confidence": float(result["yolo_conf"]),
            "class_id": int(result["yolo_cls"]),
        },
        "context_crop_config": DEFAULT_CONTEXT_CROP_CONFIG,
        "targets": targets_manifest,
    }

    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return bundle_dir


def run_single_image_pipeline(selected_path, yolo_model, dev, imgsz, stride,
                              conf_thres, iou_thres, cnn_classifier, output_dir=OUTPUT_DIR):
    """Tek görüntü için YOLO -> crop -> CNN -> annotate akışını çalıştırır."""
    img_bgr = cv2.imread(str(selected_path))
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    detections = yolo_detect(yolo_model, img_bgr, dev, imgsz, stride, conf_thres, iou_thres)
    yolo_vis = draw_yolo_boxes(img_bgr, detections)

    crops = []
    valid_detections = []
    for det in detections:
        crop, _ = crop_with_context(
            img_bgr,
            (det["x1"], det["y1"], det["x2"], det["y2"]),
            **DEFAULT_CONTEXT_CROP_CONFIG,
        )
        if crop.size == 0:
            continue
        crops.append(crop)
        valid_detections.append(det)

    crops_rgb = [bgr_to_rgb(crop) for crop in crops]
    crops_pil = [Image.fromarray(crop_rgb) for crop_rgb in crops_rgb]
    batch_results = cnn_classifier.predict_batch(crops_pil, return_proba=True) if crops_pil else []

    labels_map = {0: "no_helmet", 1: "helmet"}
    results = []
    for crop_rgb, det, br in zip(crops_rgb, valid_detections, batch_results):
        label = labels_map[br["prediction"]]
        results.append({
            "x1": det["x1"],
            "y1": det["y1"],
            "x2": det["x2"],
            "y2": det["y2"],
            "yolo_conf": det["conf"],
            "yolo_cls": det["yolo_cls"],
            "label": label,
            "confidence": br["confidence"],
            "prediction": br["prediction"],
            "proba_helmet": br["probabilities"]["helmet"],
            "proba_no_helmet": br["probabilities"]["no_helmet"],
        })

    final_img = draw_final(img_bgr, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / selected_path.name
    cv2.imwrite(str(save_path), final_img)

    return {
        "image_name": selected_path.name,
        "image_path": str(selected_path),
        "height": h,
        "width": w,
        "image_rgb": bgr_to_rgb(img_bgr),
        "yolo_vis_rgb": bgr_to_rgb(yolo_vis),
        "crops_rgb": crops_rgb,
        "detections": detections,
        "results": results,
        "final_rgb": bgr_to_rgb(final_img),
        "save_path": str(save_path),
    }


def get_gradcam_layers(classifier):
    """Katman adlarını model nesnelerine map eder."""
    return {
        "Stage 3 (high-level)": classifier.model.cbam3,
        "FPN P5 (coarse)": classifier.model.fpn_cbam3,
        "FPN P4 (mid)": classifier.model.fpn_cbam2,
        "FPN P3 (fine)": classifier.model.fpn_cbam1,
    }


def render_activation_analysis(cnn_classifier, pipeline_data):
    """Seçilen crop için Grad-CAM aktivasyonlarını gösterir."""
    results = pipeline_data["results"]
    if not results:
        st.warning("Aktivasyon analizi için sınıflandırılmış crop bulunamadı.")
        return

    st.markdown("---")
    st.header("Adım 6: Aktivasyon Analizi")
    st.caption(
        "Sıcak bölgeler seçilen hedef sınıf skoruna en çok katkı veren alanları gösterir. "
        "Yanlış `no_helmet` kararlarında `helmet` hedefini seçip karşılaştırman faydalı olur."
    )

    detection_idx = st.selectbox(
        "Analiz edilecek crop",
        options=list(range(len(results))),
        format_func=lambda idx: (
            f"Det {idx + 1} | {results[idx]['label']} | "
            f"CNN {results[idx]['confidence']:.1%} | YOLO {results[idx]['yolo_conf']:.1%}"
        ),
        key=f"activation_det_{pipeline_data['image_name']}",
    )

    selected_result = results[detection_idx]
    target_mode = st.radio(
        "Hedef skor",
        options=["prediction", "helmet", "no_helmet"],
        format_func=lambda mode: {
            "prediction": f"Tahmin edilen sınıf ({selected_result['label']})",
            "helmet": "helmet skoruna bak",
            "no_helmet": "no_helmet skoruna bak",
        }[mode],
        horizontal=True,
        key=f"activation_target_{pipeline_data['image_name']}_{detection_idx}",
    )

    if target_mode == "prediction":
        target_class = None
        target_label = selected_result["label"]
    else:
        target_class = 1 if target_mode == "helmet" else 0
        target_label = target_mode

    crop_pil = Image.fromarray(pipeline_data["crops_rgb"][detection_idx])
    layers = get_gradcam_layers(cnn_classifier)

    with st.spinner("Grad-CAM üretiliyor..."):
        gradcam_views = {
            name: cnn_classifier.gradcam(crop_pil, target_class=target_class, target_layer=layer)
            for name, layer in layers.items()
        }

    info_col, metric_col = st.columns([1.2, 1.0])
    with info_col:
        st.image(pipeline_data["crops_rgb"][detection_idx], caption="Seçilen crop", width="stretch")
    with metric_col:
        st.markdown(
            f"**Tahmin:** `{selected_result['label']}`  \n"
            f"**Hedef skor:** `{target_label}`  \n"
            f"**CNN confidence:** `{selected_result['confidence']:.1%}`"
        )
        st.metric("helmet olasılığı", f"{selected_result['proba_helmet']:.1%}")
        st.metric("no_helmet olasılığı", f"{selected_result['proba_no_helmet']:.1%}")
        st.caption(
            f"BBox: ({selected_result['x1']}, {selected_result['y1']}) - "
            f"({selected_result['x2']}, {selected_result['y2']})"
        )

    save_col, compare_col, info_col = st.columns([1.0, 1.0, 1.8])
    save_key = (
        f"save_activation_{pipeline_data['image_name']}_{detection_idx}_{target_label}"
    )
    with save_col:
        if st.button("Aktivasyonları Kaydet", key=save_key, type="primary"):
            saved_dir = save_activation_bundle(
                pipeline_data=pipeline_data,
                detection_idx=detection_idx,
                target_label=target_label,
                gradcam_views=gradcam_views,
            )
            st.success(f"Kaydedildi: {saved_dir}")
    comparison_key = f"save_activation_compare_{pipeline_data['image_name']}_{detection_idx}"
    with compare_col:
        if st.button("İki Sınıfı da Kaydet", key=comparison_key):
            with st.spinner("helmet + no_helmet karşılaştırması hazırlanıyor..."):
                comparison_targets = {}
                for mode in ("helmet", "no_helmet"):
                    label, target_gradcam_views = get_target_gradcam_views(
                        cnn_classifier,
                        crop_pil,
                        target_mode=mode,
                        layers=layers,
                        predicted_label=selected_result["label"],
                    )
                    comparison_targets[label] = target_gradcam_views

                saved_dir = save_activation_comparison_bundle(
                    pipeline_data=pipeline_data,
                    detection_idx=detection_idx,
                    target_views=comparison_targets,
                )
            st.success(f"Karşılaştırma paketi kaydedildi: {saved_dir}")
    with info_col:
        st.caption(
            "Kayıt klasörü: "
            f"{ACTIVATION_OUTPUT_DIR / Path(pipeline_data['image_name']).stem}"
        )
        st.caption(
            "Kaydedilen pakette overlay PNG'ler, ham heatmap `.npy` dosyaları ve "
            "`metadata.json` olacak."
        )
        st.caption(
            "Karşılaştırmalı kayıt klasörü: "
            f"{ACTIVATION_COMPARISON_OUTPUT_DIR / Path(pipeline_data['image_name']).stem}"
        )

    layer_items = list(gradcam_views.items())
    for start in range(0, len(layer_items), 2):
        cols = st.columns(2)
        for col, (layer_name, cam_result) in zip(cols, layer_items[start:start + 2]):
            with col:
                st.image(cam_result["overlay"], caption=layer_name, width="stretch")
                st.caption(
                    f"Pred={cam_result['prediction']} | "
                    f"helmet={cam_result['probabilities']['helmet']:.1%} | "
                    f"no_helmet={cam_result['probabilities']['no_helmet']:.1%}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Helmet Detection Pipeline", layout="wide")
    st.title("Helmet Detection Pipeline")
    st.markdown("**YOLO Tespit → Kırpma → CNN Sınıflandırma → Sonuç**")

    if "single_image_pipeline" not in st.session_state:
        st.session_state.single_image_pipeline = None

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
        pipeline_data = run_single_image_pipeline(
            selected_path=selected_path,
            yolo_model=yolo_model,
            dev=dev,
            imgsz=imgsz,
            stride=stride,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            cnn_classifier=cnn_classifier,
        )
        if pipeline_data is None:
            st.error("Görüntü okunamadı!")
            return
        st.session_state.single_image_pipeline = pipeline_data

    pipeline_data = st.session_state.single_image_pipeline
    if pipeline_data and pipeline_data["image_name"] != selected_name:
        st.info("Seçtiğin yeni görüntü için tekrar `Pipeline Çalıştır` demen gerekiyor.")
        pipeline_data = None

    if pipeline_data:
        h, w = pipeline_data["height"], pipeline_data["width"]
        # ── STEP 1: Original ─────────────────────────────────────────────
        st.markdown("---")
        st.header("Adım 1: Orijinal Görüntü")
        st.image(pipeline_data["image_rgb"], caption=f"{selected_name} ({w}x{h})", width="stretch")

        # ── STEP 2: YOLO Detection ───────────────────────────────────────
        st.markdown("---")
        st.header("Adım 2: YOLO Tespitleri")
        detections = pipeline_data["detections"]
        st.success(f"{len(detections)} tespit bulundu")
        st.image(
            pipeline_data["yolo_vis_rgb"],
            caption="YOLO tespitleri (sınıf etiketsiz, sadece bbox)",
            width="stretch",
        )

        if not detections:
            st.warning("Hiç tespit bulunamadı. Confidence threshold'u düşürmeyi deneyin.")
            return

        # ── STEP 3: Crop Detections ──────────────────────────────────────
        st.markdown("---")
        st.header("Adım 3: Kırpılmış Tespitler")

        crops = pipeline_data["crops_rgb"]

        cols = st.columns(min(len(crops), 6))
        for i, crop in enumerate(crops):
            with cols[i % len(cols)]:
                st.image(crop, caption=f"Det {i+1}", width="stretch")

        # ── STEP 4: CNN Classification (batch) ──────────────────────────
        st.markdown("---")
        st.header("Adım 4: CNN Sınıflandırma")
        results = pipeline_data["results"]
        cols = st.columns(min(len(crops), 6))
        for i, (crop, result) in enumerate(zip(crops, results)):
            with cols[i % len(cols)]:
                emoji = "✅" if result["label"] == "helmet" else "❌"
                color = "green" if result["label"] == "helmet" else "red"
                st.image(crop, width="stretch")
                st.markdown(
                    f'{emoji} **:{color}[{result["label"]}]** '
                    f'({result["confidence"]:.0%})'
                )
                st.caption(
                    f'Helmet: {result["proba_helmet"]:.1%} | '
                    f'No helmet: {result["proba_no_helmet"]:.1%}'
                )

        # ── STEP 5: Final Result ─────────────────────────────────────────
        st.markdown("---")
        st.header("Adım 5: Sonuç")
        st.image(pipeline_data["final_rgb"], caption="Final sonuç", width="stretch")
        st.success(f"Kaydedildi: {pipeline_data['save_path']}")

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

        render_activation_analysis(cnn_classifier, pipeline_data)

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
