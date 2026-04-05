#!/usr/bin/env python3
"""
Batch export of dual-target Grad-CAM bundles for pipeline detections.

Her detection için hem `helmet` hem `no_helmet` hedef aktivasyonlarını üretir,
ayrı klasöre kaydeder ve özet rapor çıkarır.
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

from streamlit_pipeline import (
    INPUT_DIR,
    PROJECT_ROOT,
    get_gradcam_layers,
    get_target_gradcam_views,
    load_cnn_model,
    load_yolo_model,
    run_single_image_pipeline,
    save_activation_comparison_bundle,
)


def build_report(records, images_processed, output_root, pipeline_output_dir):
    """JSON + markdown özet raporu üret."""
    prediction_counts = Counter(record["prediction"]["label"] for record in records)
    high_conf_no_helmet = sorted(
        [r for r in records if r["prediction"]["label"] == "no_helmet"],
        key=lambda item: item["prediction"]["confidence"],
        reverse=True,
    )
    low_conf_helmet = sorted(
        [r for r in records if r["prediction"]["label"] == "helmet"],
        key=lambda item: item["prediction"]["confidence"],
    )
    borderline = sorted(
        [r for r in records if 0.45 <= r["prediction"]["confidence"] <= 0.60],
        key=lambda item: item["prediction"]["confidence"],
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "images_processed": images_processed,
        "pipeline_output_dir": str(pipeline_output_dir),
        "activation_output_dir": str(output_root),
        "total_detections": len(records),
        "prediction_counts": dict(prediction_counts),
        "high_conf_no_helmet": high_conf_no_helmet[:10],
        "low_conf_helmet": low_conf_helmet[:10],
        "borderline_predictions": borderline[:10],
        "records": records,
    }

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# Dual-Target Activation Export Report",
        "",
        f"- Images processed: `{images_processed}`",
        f"- Total detections: `{len(records)}`",
        f"- Pipeline outputs: `{pipeline_output_dir}`",
        f"- Activation bundles: `{output_root}`",
        "",
        "## Prediction Counts",
        "",
    ]
    for label, count in sorted(prediction_counts.items()):
        lines.append(f"- `{label}`: `{count}`")

    def append_section(title, items):
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("- None")
            return
        for item in items:
            pred = item["prediction"]
            probs = pred["probabilities"]
            lines.append(
                "- "
                f"{item['image_name']} det{item['detection_index'] + 1}: "
                f"`{pred['label']}` {pred['confidence']:.1%} | "
                f"helmet={probs['helmet']:.1%} | no_helmet={probs['no_helmet']:.1%} | "
                f"[bundle]({item['bundle_dir']})"
            )

    append_section("High-Confidence no_helmet", high_conf_no_helmet[:10])
    append_section("Low-Confidence helmet", low_conf_helmet[:10])
    append_section("Borderline Predictions", borderline[:10])

    with open(output_root / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="YOLO->CNN pipeline için dual-target Grad-CAM export"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="İşlenecek test görüntü klasörü",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "050426_results_dual_export",
        help="Final annotated pipeline çıktılarının yazılacağı klasör",
    )
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "activation_debug_dual_target_batch",
        help="Dual-target aktivasyon klasörü",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Belirli dosyaları seçmek için opsiyonel görüntü listesi",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.activation_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in image_extensions and (args.images is None or p.name in args.images)
    )

    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    print("Loading models...")
    yolo_model, dev, imgsz, stride = load_yolo_model()
    cnn_classifier = load_cnn_model()
    layers = get_gradcam_layers(cnn_classifier)

    records = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing {image_path.name}")
        pipeline_data = run_single_image_pipeline(
            selected_path=image_path,
            yolo_model=yolo_model,
            dev=dev,
            imgsz=imgsz,
            stride=stride,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            cnn_classifier=cnn_classifier,
            output_dir=args.results_dir,
        )
        if pipeline_data is None:
            print(f"  ! Skipping unreadable image: {image_path}")
            continue

        for detection_idx, result in enumerate(pipeline_data["results"]):
            crop_pil = Image.fromarray(pipeline_data["crops_rgb"][detection_idx])
            target_views = {}
            for target_mode in ("helmet", "no_helmet"):
                target_label, gradcam_views = get_target_gradcam_views(
                    cnn_classifier,
                    crop_pil,
                    target_mode=target_mode,
                    layers=layers,
                    predicted_label=result["label"],
                )
                target_views[target_label] = gradcam_views

            bundle_dir = save_activation_comparison_bundle(
                pipeline_data=pipeline_data,
                detection_idx=detection_idx,
                target_views=target_views,
                output_root=args.activation_dir,
            )

            records.append({
                "image_name": pipeline_data["image_name"],
                "image_path": pipeline_data["image_path"],
                "detection_index": detection_idx,
                "bbox": {
                    "x1": result["x1"],
                    "y1": result["y1"],
                    "x2": result["x2"],
                    "y2": result["y2"],
                },
                "prediction": {
                    "label": result["label"],
                    "class_id": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": {
                        "helmet": result["proba_helmet"],
                        "no_helmet": result["proba_no_helmet"],
                    },
                },
                "yolo": {
                    "confidence": result["yolo_conf"],
                    "class_id": result["yolo_cls"],
                },
                "bundle_dir": str(bundle_dir),
            })

    summary = build_report(
        records=records,
        images_processed=len(image_paths),
        output_root=args.activation_dir,
        pipeline_output_dir=args.results_dir,
    )

    print("\nExport finished.")
    print(f"  Images:     {summary['images_processed']}")
    print(f"  Detections: {summary['total_detections']}")
    print(f"  Results:    {args.results_dir}")
    print(f"  Activations:{args.activation_dir}")
    print(f"  Report:     {args.activation_dir / 'report.md'}")


if __name__ == "__main__":
    main()
