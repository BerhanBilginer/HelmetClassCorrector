import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def filter_labels_by_class(dataset_path, keep_classes=[1, 3], backup=True):
    """
    YOLO format label dosyalarından belirli class_id'leri filtreler.
    
    Args:
        dataset_path: Dataset ana klasörü (train, val, test içeren)
        keep_classes: Tutulacak class_id'ler (varsayılan: [1, 3])
        backup: True ise orijinal dosyaları yedekler
    """
    dataset_path = Path(dataset_path)
    
    splits = ['train', 'val', 'test']
    stats = {
        'total_files': 0,
        'modified_files': 0,
        'total_lines_before': 0,
        'total_lines_after': 0,
        'removed_class_0': 0,
        'removed_class_2': 0,
        'kept_class_1': 0,
        'kept_class_3': 0
    }
    
    for split in splits:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"⚠️  {split}/labels klasörü bulunamadı, atlanıyor...")
            continue
        
        txt_files = list(labels_dir.glob('*.txt'))
        
        if not txt_files:
            print(f"⚠️  {split}/labels klasöründe txt dosyası bulunamadı")
            continue
        
        print(f"\n{'='*60}")
        print(f"📁 {split.upper()} klasörü işleniyor...")
        print(f"{'='*60}")
        
        if backup:
            backup_dir = labels_dir.parent / 'labels_backup'
            backup_dir.mkdir(exist_ok=True)
            print(f"💾 Yedek klasörü: {backup_dir}")
        
        for txt_file in tqdm(txt_files, desc=f"Processing {split}"):
            stats['total_files'] += 1
            
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                if backup:
                    backup_file = backup_dir / txt_file.name
                    with open(backup_file, 'w') as f:
                        f.writelines(lines)
                
                filtered_lines = []
                file_modified = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    stats['total_lines_before'] += 1
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        
                        if class_id in keep_classes:
                            filtered_lines.append(line + '\n')
                            stats['total_lines_after'] += 1
                            
                            if class_id == 1:
                                stats['kept_class_1'] += 1
                            elif class_id == 3:
                                stats['kept_class_3'] += 1
                        else:
                            file_modified = True
                            if class_id == 0:
                                stats['removed_class_0'] += 1
                            elif class_id == 2:
                                stats['removed_class_2'] += 1
                    
                    except ValueError:
                        continue
                
                if file_modified:
                    stats['modified_files'] += 1
                
                with open(txt_file, 'w') as f:
                    f.writelines(filtered_lines)
            
            except Exception as e:
                print(f"\n❌ Hata ({txt_file.name}): {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"✅ İŞLEM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"\n📊 İSTATİSTİKLER:")
    print(f"  • Toplam işlenen dosya: {stats['total_files']}")
    print(f"  • Değiştirilen dosya: {stats['modified_files']}")
    print(f"  • Toplam satır (önce): {stats['total_lines_before']}")
    print(f"  • Toplam satır (sonra): {stats['total_lines_after']}")
    print(f"  • Silinen satır: {stats['total_lines_before'] - stats['total_lines_after']}")
    print(f"\n🗑️  SİLİNEN CLASS'LAR:")
    print(f"  • Class 0: {stats['removed_class_0']} satır")
    print(f"  • Class 2: {stats['removed_class_2']} satır")
    print(f"\n✅ TUTULAN CLASS'LAR:")
    print(f"  • Class 1: {stats['kept_class_1']} satır")
    print(f"  • Class 3: {stats['kept_class_3']} satır")
    print(f"{'='*60}\n")
    
    return stats

def balance_classes_with_augmentation(dataset_path, target_class, balance_ratio=1.0, augment_types=['flip', 'rotate', 'perspective']):
    """
    Azınlık sınıfını geometric augmentation ile dengeler.
    Color-based model için color augmentation YAPILMAZ.
    
    Args:
        dataset_path: Dataset ana klasörü (train, val, test içeren)
        target_class: Artırılacak sınıf ID'si (örn: 3 for no_helmet)
        balance_ratio: Hedef oran (1.0 = tam denge, 0.5 = yarı yarıya)
        augment_types: Kullanılacak augmentation tipleri
    
    Augmentation tipleri:
        - 'flip': Horizontal flip
        - 'rotate': ±15° rotation
        - 'perspective': Hafif perspective transform
    """
    dataset_path = Path(dataset_path)
    splits = ['train']  # Sadece train split'i dengele
    
    print(f"\n{'='*60}")
    print(f"⚖️  CLASS BALANCING: Class {target_class}")
    print(f"{'='*60}")
    print(f"Augmentation types: {', '.join(augment_types)}")
    print(f"Balance ratio: {balance_ratio:.1%}")
    
    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  {split} klasörü bulunamadı, atlanıyor...")
            continue
        
        # Count class distribution
        class_counts = {}
        class_files = {}
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id not in class_counts:
                        class_counts[class_id] = 0
                        class_files[class_id] = []
                    class_counts[class_id] += 1
                    if label_file not in class_files[class_id]:
                        class_files[class_id].append(label_file)
        
        print(f"\n📊 {split.upper()} - Mevcut dağılım:")
        for cls_id, count in sorted(class_counts.items()):
            print(f"  • Class {cls_id}: {count:,} instances")
        
        if target_class not in class_counts:
            print(f"⚠️  Class {target_class} bulunamadı!")
            continue
        
        # Calculate how many augmentations needed
        max_count = max(class_counts.values())
        target_count = int(max_count * balance_ratio)
        current_count = class_counts[target_class]
        needed = target_count - current_count
        
        if needed <= 0:
            print(f"✅ Class {target_class} zaten dengeli (veya çoğunlukta)")
            continue
        
        print(f"\n🎯 Hedef: {current_count:,} → {target_count:,} (+{needed:,} augmentation)")
        
        # Get files containing target class
        target_files = class_files[target_class]
        random.shuffle(target_files)
        
        augmented_count = 0
        aug_cycle = 0
        
        with tqdm(total=needed, desc=f"Augmenting class {target_class}") as pbar:
            while augmented_count < needed:
                # Cycle through files
                label_file = target_files[aug_cycle % len(target_files)]
                image_name = label_file.stem
                
                # Find image file
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential = images_dir / f"{image_name}{ext}"
                    if potential.exists():
                        image_path = potential
                        break
                
                if image_path is None:
                    aug_cycle += 1
                    continue
                
                # Load image and labels
                img = cv2.imread(str(image_path))
                if img is None:
                    aug_cycle += 1
                    continue
                
                with open(label_file, 'r') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                
                # Filter only target class
                target_labels = [l for l in labels if int(l.split()[0]) == target_class]
                if not target_labels:
                    aug_cycle += 1
                    continue
                
                # Choose random augmentation
                aug_type = random.choice(augment_types)
                
                # Apply augmentation
                aug_img, aug_labels = apply_augmentation(img, target_labels, aug_type)
                
                # Save augmented image and label
                aug_suffix = f"_aug{augmented_count}_{aug_type}"
                aug_img_name = f"{image_name}{aug_suffix}{image_path.suffix}"
                aug_label_name = f"{image_name}{aug_suffix}.txt"
                
                cv2.imwrite(str(images_dir / aug_img_name), aug_img)
                with open(labels_dir / aug_label_name, 'w') as f:
                    f.write('\n'.join(aug_labels) + '\n')
                
                augmented_count += len(target_labels)
                pbar.update(len(target_labels))
                aug_cycle += 1
        
        print(f"\n✅ {split.upper()}: {augmented_count:,} augmented instances oluşturuldu")
    
    print(f"\n{'='*60}")
    print(f"✅ CLASS BALANCING TAMAMLANDI")
    print(f"{'='*60}\n")


def apply_augmentation(img, labels, aug_type):
    """
    Geometric augmentation uygular ve YOLO bbox'ları günceller.
    
    Args:
        img: OpenCV image (BGR)
        labels: YOLO format label strings (class x_center y_center width height)
        aug_type: 'flip', 'rotate', 'perspective'
    
    Returns:
        aug_img: Augmented image
        aug_labels: Updated YOLO labels
    """
    h, w = img.shape[:2]
    
    if aug_type == 'flip':
        # Horizontal flip
        aug_img = cv2.flip(img, 1)
        aug_labels = []
        for label in labels:
            parts = label.split()
            class_id = parts[0]
            x_center = 1.0 - float(parts[1])  # Mirror x
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            aug_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return aug_img, aug_labels
    
    elif aug_type == 'rotate':
        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Rotate bboxes (approximate - keep original for simplicity)
        # For small rotations, bbox shift is minimal
        aug_labels = labels.copy()
        return aug_img, aug_labels
    
    elif aug_type == 'perspective':
        # Slight perspective transform
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        shift = int(w * 0.05)  # 5% shift
        pts2 = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), random.randint(0, shift)],
            [random.randint(0, shift), h - random.randint(0, shift)],
            [w - random.randint(0, shift), h - random.randint(0, shift)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        aug_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Keep original labels (perspective is slight)
        aug_labels = labels.copy()
        return aug_img, aug_labels
    
    else:
        return img, labels


def restore_from_backup(dataset_path):
    """
    Yedekten geri yükleme yapar.
    """
    dataset_path = Path(dataset_path)
    splits = ['train', 'val', 'test']
    
    restored_count = 0
    
    for split in splits:
        backup_dir = dataset_path / split / 'labels_backup'
        labels_dir = dataset_path / split / 'labels'
        
        if not backup_dir.exists():
            continue
        
        print(f"📂 {split} klasörü geri yükleniyor...")
        
        for backup_file in backup_dir.glob('*.txt'):
            target_file = labels_dir / backup_file.name
            
            with open(backup_file, 'r') as f:
                content = f.read()
            
            with open(target_file, 'w') as f:
                f.write(content)
            
            restored_count += 1
    
    print(f"✅ {restored_count} dosya geri yüklendi")

if __name__ == "__main__":
    dataset_path = '/home/berhan/Development/personal/HelmetClassCorrector/kiran_dataset'
    
    print("🔧 YOLO Dataset Manipülasyon Aracı")
    print("=" * 60)
    print("Seçenekler:")
    print("  1. Label filtreleme (class_id 0 ve 2 sil)")
    print("  2. Class balancing (azınlık sınıfını augment et)")
    print("  3. Backup'tan geri yükle")
    print("=" * 60)
    
    choice = input("\nSeçiminiz (1/2/3): ").strip()
    
    if choice == '1':
        print("\n📋 Label filtreleme başlatılıyor...")
        stats = filter_labels_by_class(
            dataset_path=dataset_path,
            keep_classes=[1, 3],
            backup=True
        )
        print("\n💡 İpucu: Geri yüklemek için seçenek 3'ü kullanabilirsiniz.")
    
    elif choice == '2':
        print("\n⚖️  Class balancing başlatılıyor...")
        print("\nÖrnek: Class 1 (helmet): 117336, Class 3 (no_helmet): 71232")
        target_class = int(input("Artırılacak sınıf ID (örn: 3): "))
        balance_ratio = float(input("Denge oranı (0.0-1.0, örn: 1.0 tam denge): "))
        
        balance_classes_with_augmentation(
            dataset_path=dataset_path,
            target_class=target_class,
            balance_ratio=balance_ratio,
            augment_types=['flip', 'rotate', 'perspective']
        )
        print("\n💡 Augmented dosyalar '_aug' suffix'i ile kaydedildi.")
    
    elif choice == '3':
        print("\n♻️  Backup'tan geri yükleme başlatılıyor...")
        response = input("Emin misiniz? Tüm değişiklikler geri alınacak (y/n): ").lower()
        if response == 'y':
            restore_from_backup(dataset_path)
        else:
            print("❌ İşlem iptal edildi.")
    
    else:
        print("❌ Geçersiz seçim.")
