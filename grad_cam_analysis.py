#!/usr/bin/env python3
"""
Grad-CAM Analiz Aracı — Model Debugging

Kullanım:
  # Tek görüntü analizi
  python grad_cam_analysis.py --model models/trained/helmet_classifier_v5.pth \
                               --image test/crops/dark_helmet.png

  # Klasör analizi (tüm yanlış tahminleri bul)
  python grad_cam_analysis.py --model models/trained/helmet_classifier_v5.pth \
                               --dataset dataset --split test

  # v4 model ile karşılaştırma
  python grad_cam_analysis.py --model models/trained/helmet_classifier_v4.pth \
                               --image test/crops/dark_helmet.png

Çıktılar:
  - gradcam_outputs/  klasörüne heatmap overlay'ler kaydedilir
  - Yanlış tahminler ayrı klasöre toplanır
  - Confidence dağılım grafiği üretilir
"""

import argparse
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch


def analyze_single_image(classifier, image_path, output_dir, use_tta=False):
    """
    Tek bir görüntü için Grad-CAM analizi yapar.

    Üretir:
      1. 4 katmanlı Grad-CAM karşılaştırma (Stage3, FPN P5/P4/P3)
      2. TTA vs normal tahmin karşılaştırması
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_img = Image.open(image_path).convert('RGB')

    # Normal prediction
    result_normal = classifier.predict(image_path, return_proba=True)
    label_normal = 'helmet' if result_normal['prediction'] == 1 else 'no_helmet'

    print(f"\n📸 {image_path.name}")
    print(f"   Normal:  {label_normal} ({result_normal['confidence']:.1%})")
    print(f"   Probs:   helmet={result_normal['probabilities']['helmet']:.3f} | "
          f"no_helmet={result_normal['probabilities']['no_helmet']:.3f}")

    # TTA prediction
    if use_tta:
        result_tta = classifier.predict_tta(image_path, return_proba=True)
        label_tta = 'helmet' if result_tta['prediction'] == 1 else 'no_helmet'
        print(f"   TTA:     {label_tta} ({result_tta['confidence']:.1%}) "
              f"[{result_tta['tta_views']} views]")
        print(f"   Probs:   helmet={result_tta['probabilities']['helmet']:.3f} | "
              f"no_helmet={result_tta['probabilities']['no_helmet']:.3f}")

    # Grad-CAM comparison across layers
    print("   Grad-CAM üretiliyor...")
    save_path = output_dir / f"gradcam_{image_path.stem}.png"
    classifier.gradcam_comparison(pil_img, save_path=str(save_path))

    # Tek katman detaylı heatmap (FPN P5 — en anlamlı genelde)
    result_cam = classifier.gradcam(pil_img)
    overlay_path = output_dir / f"overlay_{image_path.stem}.png"
    overlay_rgb = result_cam['overlay']
    Image.fromarray(overlay_rgb).save(str(overlay_path))

    print(f"   ✓ Kaydedildi: {save_path.name}")
    return result_normal, result_cam


def analyze_dataset(classifier, dataset_dir, split='test', output_dir='gradcam_outputs',
                    save_errors_only=True, use_tta=False):
    """
    Dataset üzerinde Grad-CAM analizi — yanlış tahminleri bulur ve görselleştirir.

    Args:
        classifier: CNNClassifier instance
        dataset_dir: Dataset ana klasörü
        split: 'train', 'val', veya 'test'
        output_dir: Çıktı klasörü
        save_errors_only: True → sadece yanlış tahminler için Grad-CAM üret
        use_tta: True → TTA ile tekrar tahmin yap
    """
    dataset_dir = Path(dataset_dir) / split
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_dir = output_dir / 'errors'
    errors_dir.mkdir(exist_ok=True)

    stats = {
        'total': 0,
        'correct': 0,
        'errors': [],
        'confidences_correct': [],
        'confidences_wrong': [],
        'tta_recoveries': 0,
    }

    classes = {'helmet': 1, 'no_helmet': 0}

    for class_name, class_id in classes.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        print(f"\n📁 {class_name}: {len(images)} görüntü")

        for img_path in tqdm(images, desc=f"  {class_name}"):
            stats['total'] += 1

            result = classifier.predict(img_path, return_proba=True)
            pred = result['prediction']
            conf = result['confidence']

            if pred == class_id:
                stats['correct'] += 1
                stats['confidences_correct'].append(conf)

                if not save_errors_only:
                    # Grad-CAM sadece istenmişse
                    cam_result = classifier.gradcam(img_path)
                    save_cam = errors_dir / f"correct_{class_name}_{img_path.name}"
                    Image.fromarray(cam_result['overlay']).save(str(save_cam))
            else:
                stats['confidences_wrong'].append(conf)

                # TTA ile tekrar dene
                tta_result = None
                tta_recovered = False
                if use_tta:
                    tta_result = classifier.predict_tta(img_path, return_proba=True)
                    if tta_result['prediction'] == class_id:
                        tta_recovered = True
                        stats['tta_recoveries'] += 1

                # Grad-CAM üret (yanlış tahminler için her zaman)
                cam_result = classifier.gradcam(img_path)
                err_prefix = "tta_fixed" if tta_recovered else "error"
                save_cam = errors_dir / f"{err_prefix}_{class_name}_{img_path.stem}.png"
                Image.fromarray(cam_result['overlay']).save(str(save_cam))

                stats['errors'].append({
                    'image': str(img_path),
                    'true_label': class_name,
                    'pred_label': 'helmet' if pred == 1 else 'no_helmet',
                    'confidence': conf,
                    'tta_recovered': tta_recovered,
                    'tta_conf': tta_result['confidence'] if tta_result else None,
                })

    # Summary
    accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"\n{'='*60}")
    print(f"📊 ANALIZ SONUÇLARI ({split})")
    print(f"{'='*60}")
    print(f"  Accuracy:         {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    print(f"  Yanlış tahmin:    {len(stats['errors'])}")
    if use_tta:
        print(f"  TTA ile düzelen:  {stats['tta_recoveries']}")

    if stats['confidences_correct']:
        print(f"\n  Doğru tahmin confidence:")
        print(f"    Mean: {np.mean(stats['confidences_correct']):.3f}")
        print(f"    Min:  {np.min(stats['confidences_correct']):.3f}")

    if stats['confidences_wrong']:
        print(f"\n  Yanlış tahmin confidence:")
        print(f"    Mean: {np.mean(stats['confidences_wrong']):.3f}")
        print(f"    Max:  {np.max(stats['confidences_wrong']):.3f} ← overconfident!")

    print(f"\n  Grad-CAM çıktıları: {errors_dir}")
    print(f"{'='*60}")

    # Confidence dağılım grafiği
    plot_confidence_distribution(stats, output_dir)

    # Hata listesi
    if stats['errors']:
        print(f"\n❌ En yüksek confidence'lı hatalar:")
        sorted_errors = sorted(stats['errors'], key=lambda x: x['confidence'], reverse=True)
        for e in sorted_errors[:10]:
            tta_tag = " → TTA ✅" if e.get('tta_recovered') else ""
            print(f"   {Path(e['image']).name}: "
                  f"true={e['true_label']}, pred={e['pred_label']} "
                  f"({e['confidence']:.1%}){tta_tag}")

    return stats


def plot_confidence_distribution(stats, output_dir):
    """Doğru vs yanlış tahminlerin confidence dağılımını çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if stats['confidences_correct']:
        axes[0].hist(stats['confidences_correct'], bins=50, alpha=0.7,
                     color='green', label='Correct')
    if stats['confidences_wrong']:
        axes[0].hist(stats['confidences_wrong'], bins=20, alpha=0.7,
                     color='red', label='Wrong')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Calibration plot — idealde 45° çizgi
    if stats['confidences_correct'] and stats['confidences_wrong']:
        all_confs = stats['confidences_correct'] + stats['confidences_wrong']
        all_correct = [1] * len(stats['confidences_correct']) + [0] * len(stats['confidences_wrong'])

        bins = np.linspace(0.5, 1.0, 11)
        bin_indices = np.digitize(all_confs, bins)

        bin_accs = []
        bin_confs = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if sum(mask) > 0:
                bin_correct = [all_correct[j] for j in range(len(mask)) if mask[j]]
                bin_conf = [all_confs[j] for j in range(len(mask)) if mask[j]]
                bin_accs.append(np.mean(bin_correct))
                bin_confs.append(np.mean(bin_conf))

        axes[1].plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, label='Perfect calibration')
        axes[1].plot(bin_confs, bin_accs, 'bo-', label='Model')
        axes[1].set_xlabel('Mean Predicted Confidence')
        axes[1].set_ylabel('Actual Accuracy')
        axes[1].set_title('Calibration Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / 'confidence_analysis.png'
    plt.savefig(str(save_path), dpi=150)
    print(f"✓ Confidence analizi: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM Model Analiz Aracı',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True, help='Model dosya yolu (.pth)')
    parser.add_argument('--image', type=str, default=None, help='Tek görüntü analizi')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset klasörü')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--output-dir', type=str, default='gradcam_outputs')
    parser.add_argument('--tta', action='store_true', help='TTA ile karşılaştırma')
    parser.add_argument('--all', action='store_true',
                        help='Sadece hatalı değil, tüm görüntüler için Grad-CAM')
    args = parser.parse_args()

    if args.image is None and args.dataset is None:
        parser.error("--image veya --dataset belirtilmeli")

    # Model yükle
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.models.cnn_classifier import CNNClassifier
    classifier = CNNClassifier()
    classifier.load(args.model)
    classifier.model.eval()

    if args.image:
        analyze_single_image(
            classifier, args.image, args.output_dir, use_tta=args.tta
        )
    elif args.dataset:
        analyze_dataset(
            classifier, args.dataset, split=args.split,
            output_dir=args.output_dir,
            save_errors_only=not args.all,
            use_tta=args.tta,
        )


if __name__ == "__main__":
    main()