import cv2
import os
from pathlib import Path
from tqdm import tqdm

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

def crop_objects_from_dataset(dataset_path, output_base_path, class_mapping={1: 'helmet', 3: 'no_helmet'}):
    """
    YOLO dataset'inden objeleri crop eder ve sınıflarına göre klasörlere kaydeder.
    
    Args:
        dataset_path: YOLO dataset ana klasörü (train, val, test içeren)
        output_base_path: Crop edilmiş görüntülerin kaydedileceği ana klasör
        class_mapping: Class ID'den klasör ismine mapping (varsayılan: {1: 'helmet', 3: 'no_helmet'})
    """
    dataset_path = Path(dataset_path)
    output_base_path = Path(output_base_path)
    
    output_dirs = {}
    counters = {}
    
    for class_id, folder_name in class_mapping.items():
        output_dir = output_base_path / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[class_id] = output_dir
        counters[class_id] = get_next_number(output_dir)
    
    print(f"📁 Başlangıç numaraları:")
    for class_id, folder_name in class_mapping.items():
        print(f"  • Class {class_id} ({folder_name}): {counters[class_id]}")
    
    splits = ['train', 'val', 'test']
    total_crops = {class_id: 0 for class_id in class_mapping.keys()}
    total_images_processed = 0
    total_images_with_crops = 0
    skipped_images = 0
    
    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  {split} klasörü bulunamadı, atlanıyor...")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        
        if not label_files:
            print(f"⚠️  {split}/labels klasöründe txt dosyası bulunamadı")
            continue
        
        print(f"\n{'='*60}")
        print(f"📁 {split.upper()} klasörü işleniyor...")
        print(f"{'='*60}")
        
        for label_file in tqdm(label_files, desc=f"Processing {split}"):
            total_images_processed += 1
            
            image_name = label_file.stem
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_path = None
            for ext in image_extensions:
                potential_path = images_dir / f"{image_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
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
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    cropped_img = img[y1:y2, x1:x2]
                    
                    if cropped_img.size == 0:
                        continue
                    
                    output_dir = output_dirs[class_id]
                    output_filename = f"{counters[class_id]}.png"
                    output_path = output_dir / output_filename
                    
                    cv2.imwrite(str(output_path), cropped_img)
                    
                    counters[class_id] += 1
                    total_crops[class_id] += 1
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
    for class_id, folder_name in class_mapping.items():
        print(f"  • Class {class_id} ({folder_name}): {total_crops[class_id]} adet")
    print(f"  • Toplam: {sum(total_crops.values())} adet")
    print(f"\n📁 SON NUMARALAR:")
    for class_id, folder_name in class_mapping.items():
        print(f"  • Class {class_id} ({folder_name}): {counters[class_id] - 1}")
    print(f"{'='*60}\n")
    
    return total_crops

if __name__ == "__main__":
    dataset_path = '/home/berhan/Development/personal/HelmetClassCorrector/kiran_dataset'
    output_path = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images'
    
    print("✂️  YOLO Dataset Cropper")
    print("=" * 60)
    print("Bu script YOLO formatındaki label'ları kullanarak")
    print("objeleri crop eder ve sınıflarına göre klasörlere kaydeder.")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"\nClass Mapping:")
    print(f"  • Class 1 → {output_path}/helmet")
    print(f"  • Class 3 → {output_path}/no_helmet")
    print("=" * 60)
    
    response = input("\nDevam etmek istiyor musunuz? (y/n): ").lower()
    
    if response == 'y':
        stats = crop_objects_from_dataset(
            dataset_path=dataset_path,
            output_base_path=output_path,
            class_mapping={1: 'helmet', 3: 'no_helmet'}
        )
        
        print("\n✅ İşlem başarıyla tamamlandı!")
    else:
        print("❌ İşlem iptal edildi.")
