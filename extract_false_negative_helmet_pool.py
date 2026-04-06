#!/usr/bin/env python3
"""
Extract a review pool of likely false-negative helmet predictions.

Heuristic:
- CNN predicted `no_helmet`
- YOLO detector said `helmet`
- Helmet-target activation still overlaps the upper-helmet prior

Outputs a compact review set with copied bundles plus JSON/Markdown reports.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_LAYER_FILE = "fpn_p5__coarse_heatmap.npy"


def build_prior(height, width, sigma_x=0.40, sigma_y=0.24, center_x=0.0, center_y=-0.30):
    ys = np.linspace(-1.0, 1.0, num=height, dtype=np.float32).reshape(height, 1)
    xs = np.linspace(-1.0, 1.0, num=width, dtype=np.float32).reshape(1, width)
    sigma_x = max(float(sigma_x), 1e-6)
    sigma_y = max(float(sigma_y), 1e-6)
    prior = np.exp(
        -0.5
        * (
            ((xs - float(center_x)) / sigma_x) ** 2
            + ((ys - float(center_y)) / sigma_y) ** 2
        )
    )
    prior = prior / np.clip(prior.sum(), 1e-6, None)
    return prior


def normalized_overlap(heatmap, prior):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape={heatmap.shape}")
    heatmap = np.clip(heatmap, 0.0, None)
    if float(heatmap.sum()) <= 0:
        return 0.0
    heatmap = heatmap / float(heatmap.sum())
    return float((heatmap * prior).sum())


def load_heatmap(bundle_dir, target_name, layer_file):
    path = bundle_dir / target_name / layer_file
    if not path.exists():
        return None
    return np.load(path)


def make_safe_stem(image_name):
    return Path(image_name).stem.replace(" ", "_")


def compute_candidate_score(record, metadata, bundle_dir, layer_file, yolo_helmet_class_id):
    prediction = record["prediction"]
    yolo = record["yolo"]
    if prediction["label"] != "no_helmet":
        return None
    if int(yolo["class_id"]) != int(yolo_helmet_class_id):
        return None

    helmet_heatmap = load_heatmap(bundle_dir, "helmet", layer_file)
    no_helmet_heatmap = load_heatmap(bundle_dir, "no_helmet", layer_file)
    if helmet_heatmap is None or no_helmet_heatmap is None:
        return None

    prior = build_prior(
        helmet_heatmap.shape[0],
        helmet_heatmap.shape[1],
    )
    helmet_overlap = normalized_overlap(helmet_heatmap, prior)
    no_helmet_overlap = normalized_overlap(no_helmet_heatmap, prior)
    overlap_margin = helmet_overlap - no_helmet_overlap

    helmet_prob = float(prediction["probabilities"]["helmet"])
    no_helmet_prob = float(prediction["probabilities"]["no_helmet"])
    yolo_conf = float(yolo["confidence"])

    suspicion_score = (
        0.55 * overlap_margin
        + 0.25 * helmet_prob
        + 0.20 * yolo_conf
        + 0.10 * (1.0 - no_helmet_prob)
    )

    return {
        "image_name": record["image_name"],
        "image_path": record["image_path"],
        "detection_index": int(record["detection_index"]),
        "bundle_dir": str(bundle_dir),
        "bbox": record["bbox"],
        "prediction": prediction,
        "yolo": yolo,
        "context_crop_config": metadata.get("context_crop_config"),
        "helmet_upper_overlap": helmet_overlap,
        "no_helmet_upper_overlap": no_helmet_overlap,
        "upper_overlap_margin": overlap_margin,
        "suspicion_score": suspicion_score,
    }


def build_report(candidates, output_dir):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "total_candidates": len(candidates),
        "candidates": candidates,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# False-Negative Helmet Pool",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total candidates: `{len(candidates)}`",
        "",
        "## Ranked Candidates",
        "",
    ]
    if not candidates:
        lines.append("- None")
    else:
        for rank, candidate in enumerate(candidates, start=1):
            pred = candidate["prediction"]
            lines.extend([
                f"### {rank}. {candidate['image_name']} det{candidate['detection_index'] + 1}",
                "",
                f"- Suspicion score: `{candidate['suspicion_score']:.4f}`",
                f"- CNN prediction: `{pred['label']}` `{pred['confidence']:.1%}`",
                f"- Helmet probability: `{pred['probabilities']['helmet']:.1%}`",
                f"- YOLO confidence: `{candidate['yolo']['confidence']:.1%}`",
                f"- Helmet upper overlap: `{candidate['helmet_upper_overlap']:.4f}`",
                f"- no_helmet upper overlap: `{candidate['no_helmet_upper_overlap']:.4f}`",
                f"- Overlap margin: `{candidate['upper_overlap_margin']:.4f}`",
                f"- Bundle copy: `{candidate['review_bundle_dir']}`",
                "",
            ])

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract likely false-negative helmet review pool from activation bundles."
    )
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=Path("results/activation_debug_060426"),
        help="Dual-target activation bundle root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/false_negative_helmet_pool_060426"),
        help="Where to write review pool outputs",
    )
    parser.add_argument(
        "--layer-file",
        type=str,
        default=DEFAULT_LAYER_FILE,
        help="Heatmap file used to compute upper-helmet overlap",
    )
    parser.add_argument(
        "--yolo-helmet-class-id",
        type=int,
        default=1,
        help="YOLO class id corresponding to helmet detections",
    )
    args = parser.parse_args()

    summary_path = args.activation_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Summary not found: {summary_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_root = args.output_dir / "review_bundles"
    review_root.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = []

    for record in summary.get("records", []):
        bundle_dir = Path(record["bundle_dir"])
        metadata_path = bundle_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate = compute_candidate_score(
            record=record,
            metadata=metadata,
            bundle_dir=bundle_dir,
            layer_file=args.layer_file,
            yolo_helmet_class_id=args.yolo_helmet_class_id,
        )
        if candidate is None:
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["suspicion_score"], reverse=True)

    for rank, candidate in enumerate(candidates, start=1):
        safe_stem = make_safe_stem(candidate["image_name"])
        dest_dir = review_root / f"{rank:02d}_{safe_stem}_det{candidate['detection_index'] + 1:02d}"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(candidate["bundle_dir"], dest_dir)
        candidate["review_bundle_dir"] = str(dest_dir)
        candidate["rank"] = rank

    build_report(candidates, args.output_dir)

    print(f"Extracted {len(candidates)} false-negative helmet candidates")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
