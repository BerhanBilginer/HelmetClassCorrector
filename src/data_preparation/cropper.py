import argparse
from pathlib import Path

from src.utils.image_ops import DEFAULT_CONTEXT_CROP_CONFIG, crop_with_context

def get_next_number(output_dir):
    """
    Klasördeki mevcut dosyaların en büyük numarasını bulur ve bir sonraki numarayı döndürür.
    """
    existing_files = list(Path(output_dir).glob('*.png'))
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            num = int(f.stem)
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    YOLO formatını (normalize edilmiş) pixel koordinatlarına çevirir.
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return x1, y1, x2, y2

def crop_objects_from_dataset(
    dataset_path,
    output_base_path,
    class_mapping={1: 'helmet', 3: 'no_helmet'},
    use_dynamic_context=True,
    context_crop_config=None,
):
    """
    YOLO dataset'inden objeleri crop eder.

    Beklenen giriş yapısı:
      dataset/
        train/
          images/
          labels/
        val/
          images/
          labels/
        test/
          images/
          labels/

    Her görsel için aynı ada sahip label dosyası labels klasöründe aranır.
    Crop çıktıları split bazlı ve sınıfa göre kaydedilir:
      output_base_path/
        train/helmet, train/no_helmet
        val/helmet,   val/no_helmet
        test/helmet,  test/no_helmet
    
    Args:
        dataset_path: YOLO dataset ana klasörü (train, val, test içeren)
        output_base_path: Crop edilmiş görüntülerin kaydedileceği ana klasör
        class_mapping: Class ID'den klasör ismine mapping (varsayılan: {1: 'helmet', 3: 'no_helmet'})
        use_dynamic_context: Küçük bbox'lara ekstra çevre bilgisi ekle
        context_crop_config: Context crop ayarları sözlüğü
    """
    dataset_path = Path(dataset_path)
    output_base_path = Path(output_base_path)
    context_crop_config = context_crop_config or DEFAULT_CONTEXT_CROP_CONFIG

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required to run the cropper. "
            "Install project requirements before generating crops."
        ) from exc
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tqdm is required to run the cropper. "
            "Install project requirements before generating crops."
        ) from exc
    
    splits = ['train', 'val', 'test']
    output_dirs = {split: {} for split in splits}
    counters = {split: {} for split in splits}

    for split in splits:
        for class_id, folder_name in class_mapping.items():
            output_dir = output_base_path / split / folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[split][class_id] = output_dir
            counters[split][class_id] = get_next_number(output_dir)

    print("📁 Başlangıç numaraları:")
    for split in splits:
        print(f"  [{split}]")
        for class_id, folder_name in class_mapping.items():
            print(f"    • Class {class_id} ({folder_name}): {counters[split][class_id]}")
    if use_dynamic_context:
        print("🧭 Dynamic context crop: AÇIK")
        for key, value in context_crop_config.items():
            print(f"    • {key}: {value}")
    else:
        print("🧭 Dynamic context crop: KAPALI")

    total_crops = {
        split: {class_id: 0 for class_id in class_mapping.keys()}
        for split in splits
    }
    total_images_processed = 0
    total_images_with_crops = 0
    skipped_images = 0
    
    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  {split} klasörü bulunamadı, atlanıyor...")
            continue
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))

        if not image_files:
            print(f"⚠️  {split}/images klasöründe görsel bulunamadı")
            continue
        
        print(f"\n{'='*60}")
        print(f"📁 {split.upper()} klasörü işleniyor...")
        print(f"{'='*60}")
        
        for image_path in tqdm(sorted(image_files), desc=f"Processing {split}"):
            total_images_processed += 1

            label_file = labels_dir / f"{image_path.stem}.txt"
            if not label_file.exists():
                skipped_images += 1
                continue
            
            img = cv2.imread(str(image_path))
            if img is None:
                skipped_images += 1
                continue
            
            img_height, img_width = img.shape[:2]
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                continue
            
            has_crops = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    
                    if class_id not in class_mapping:
                        continue
                    
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    if use_dynamic_context:
                        cropped_img, _ = crop_with_context(
                            img,
                            (x1, y1, x2, y2),
                            **context_crop_config,
                        )
                    else:
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        cropped_img = img[y1:y2, x1:x2]
                    
                    if cropped_img.size == 0:
                        continue
                    
                    output_dir = output_dirs[split][class_id]
                    output_filename = f"{counters[split][class_id]}.png"
                    output_path = output_dir / output_filename
                    
                    cv2.imwrite(str(output_path), cropped_img)
                    
                    counters[split][class_id] += 1
                    total_crops[split][class_id] += 1
                    has_crops = True
                
                except (ValueError, IndexError) as e:
                    continue
            
            if has_crops:
                total_images_with_crops += 1
    
    print(f"\n{'='*60}")
    print(f"✅ CROPPING TAMAMLANDI")
    print(f"{'='*60}")
    print(f"\n📊 İSTATİSTİKLER:")
    print(f"  • Toplam işlenen görüntü: {total_images_processed}")
    print(f"  • Crop içeren görüntü: {total_images_with_crops}")
    print(f"  • Atlanan görüntü: {skipped_images}")
    print(f"\n✂️  CROP EDİLEN OBJELER:")
    grand_total = 0
    for split in splits:
        split_total = sum(total_crops[split].values())
        grand_total += split_total
        print(f"  [{split}] toplam: {split_total} adet")
        for class_id, folder_name in class_mapping.items():
            print(f"    • Class {class_id} ({folder_name}): {total_crops[split][class_id]} adet")
    print(f"  • Genel toplam: {grand_total} adet")
    print(f"\n📁 SON NUMARALAR:")
    for split in splits:
        print(f"  [{split}]")
        for class_id, folder_name in class_mapping.items():
            print(f"    • Class {class_id} ({folder_name}): {counters[split][class_id] - 1}")
    print(f"{'='*60}\n")
    
    return total_crops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop YOLO-format detections into a split/class image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="YOLO dataset root. Expected: train/val/test with images/ and labels/.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output root for cropped dataset.",
    )
    parser.add_argument(
        "--helmet-class-id",
        type=int,
        default=1,
        help="YOLO class id for helmet crops.",
    )
    parser.add_argument(
        "--no-helmet-class-id",
        type=int,
        default=3,
        help="YOLO class id for no_helmet crops.",
    )
    parser.add_argument(
        "--disable-dynamic-context",
        action="store_true",
        help="Disable extra context for small boxes and use tight bbox crops only.",
    )
    parser.add_argument(
        "--base-context-ratio",
        type=float,
        default=DEFAULT_CONTEXT_CROP_CONFIG["base_context_ratio"],
        help="Base bbox padding ratio applied on each side.",
    )
    parser.add_argument(
        "--min-context-px",
        type=int,
        default=DEFAULT_CONTEXT_CROP_CONFIG["min_context_px"],
        help="Minimum extra pixels to pad around each bbox.",
    )
    parser.add_argument(
        "--min-context-side",
        type=int,
        default=DEFAULT_CONTEXT_CROP_CONFIG["min_context_side"],
        help="Boxes shorter than this get progressively more context.",
    )
    parser.add_argument(
        "--max-context-ratio",
        type=float,
        default=DEFAULT_CONTEXT_CROP_CONFIG["max_context_ratio"],
        help="Maximum extra padding relative to bbox size.",
    )
    parser.add_argument(
        "--side-context-scale",
        type=float,
        default=DEFAULT_CONTEXT_CROP_CONFIG["side_context_scale"],
        help="Horizontal padding multiplier to keep more helmet silhouette context.",
    )
    parser.add_argument(
        "--top-context-scale",
        type=float,
        default=DEFAULT_CONTEXT_CROP_CONFIG["top_context_scale"],
        help="Top padding multiplier so crops keep more helmet crown context.",
    )
    parser.add_argument(
        "--bottom-context-scale",
        type=float,
        default=DEFAULT_CONTEXT_CROP_CONFIG["bottom_context_scale"],
        help="Bottom padding multiplier to limit extra face/torso context.",
    )
    args = parser.parse_args()

    context_crop_config = {
        "base_context_ratio": args.base_context_ratio,
        "min_context_px": args.min_context_px,
        "min_context_side": args.min_context_side,
        "max_context_ratio": args.max_context_ratio,
        "side_context_scale": args.side_context_scale,
        "top_context_scale": args.top_context_scale,
        "bottom_context_scale": args.bottom_context_scale,
    }

    print("✂️  YOLO Dataset Cropper")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output:  {args.output_path}")
    print(f"Class mapping: {args.helmet_class_id} -> helmet, {args.no_helmet_class_id} -> no_helmet")
    print(f"Dynamic context: {'OFF' if args.disable_dynamic_context else 'ON'}")
    print("=" * 60)

    crop_objects_from_dataset(
        dataset_path=args.dataset_path,
        output_base_path=args.output_path,
        class_mapping={
            args.helmet_class_id: 'helmet',
            args.no_helmet_class_id: 'no_helmet',
        },
        use_dynamic_context=not args.disable_dynamic_context,
        context_crop_config=context_crop_config,
    )

    print("\n✅ İşlem başarıyla tamamlandı!")
