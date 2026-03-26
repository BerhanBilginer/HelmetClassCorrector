import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


def count_images_in_folder(folder_path):
    """
    Klasördeki görüntü dosyalarını sayar.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    count = 0
    for ext in extensions:
        count += len(list(folder_path.glob(ext)))
    return count


def apply_geometric_augmentation(img, aug_type):
    """
    Geometric augmentation uygular (color-preserving).
    
    Args:
        img: OpenCV image (BGR)
        aug_type: 'flip', 'rotate', 'perspective'
    
    Returns:
        aug_img: Augmented image
    """
    h, w = img.shape[:2]
    
    if aug_type == 'flip':
        # Horizontal flip
        return cv2.flip(img, 1)
    
    elif aug_type == 'rotate':
        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
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
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    else:
        return img


def balance_cropped_dataset(dataset_path, balance_ratio=1.0, augment_types=['flip', 'rotate', 'perspective']):
    """
    Crop'lanmış dataset için class balancing yapar.
    
    Beklenen yapı:
      dataset/
        train/
          helmet/
          no_helmet/
    
    Args:
        dataset_path: Dataset ana klasörü
        balance_ratio: Hedef oran (1.0 = tam denge, 0.8 = %80 denge)
        augment_types: Kullanılacak augmentation tipleri
    """
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / 'train'
    
    if not train_dir.exists():
        print(f"❌ Hata: {train_dir} klasörü bulunamadı!")
        return
    
    # Class klasörlerini bul
    helmet_dir = train_dir / 'helmet'
    no_helmet_dir = train_dir / 'no_helmet'
    
    if not helmet_dir.exists() or not no_helmet_dir.exists():
        print(f"❌ Hata: helmet veya no_helmet klasörü bulunamadı!")
        return
    
    # Görüntü sayılarını say
    helmet_count = count_images_in_folder(helmet_dir)
    no_helmet_count = count_images_in_folder(no_helmet_dir)
    
    print(f"\n{'='*60}")
    print(f"📊 MEVCUT DAĞILIM")
    print(f"{'='*60}")
    print(f"  • helmet:     {helmet_count:,} images")
    print(f"  • no_helmet:  {no_helmet_count:,} images")
    print(f"  • Oran:       {max(helmet_count, no_helmet_count) / min(helmet_count, no_helmet_count):.2f}:1")
    
    # Azınlık sınıfını belirle
    if helmet_count > no_helmet_count:
        minority_class = 'no_helmet'
        minority_dir = no_helmet_dir
        minority_count = no_helmet_count
        majority_count = helmet_count
    else:
        minority_class = 'helmet'
        minority_dir = helmet_dir
        minority_count = helmet_count
        majority_count = no_helmet_count
    
    # Hedef sayıyı hesapla
    target_count = int(majority_count * balance_ratio)
    needed = target_count - minority_count
    
    if needed <= 0:
        print(f"\n✅ Dataset zaten dengeli!")
        return
    
    print(f"\n{'='*60}")
    print(f"⚖️  DENGELEME PLANI")
    print(f"{'='*60}")
    print(f"  • Azınlık sınıfı:     {minority_class}")
    print(f"  • Mevcut sayı:        {minority_count:,}")
    print(f"  • Hedef sayı:         {target_count:,}")
    print(f"  • Gerekli augment:    {needed:,}")
    print(f"  • Balance ratio:      {balance_ratio:.1%}")
    print(f"  • Augmentation types: {', '.join(augment_types)}")
    print(f"{'='*60}")
    
    # Kullanıcıdan onay al
    response = input(f"\n⚠️  {minority_class} klasörüne {needed:,} augmented görüntü eklenecek. Devam? (y/n): ").lower()
    
    if response != 'y':
        print("❌ İşlem iptal edildi.")
        return
    
    # Azınlık sınıfındaki tüm görselleri topla
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(minority_dir.glob(ext)))
    
    if not image_files:
        print(f"❌ Hata: {minority_dir} klasöründe görüntü bulunamadı!")
        return
    
    random.shuffle(image_files)
    
    print(f"\n🔄 Augmentation başlatılıyor...")
    augmented_count = 0
    aug_cycle = 0
    
    with tqdm(total=needed, desc=f"Augmenting {minority_class}") as pbar:
        while augmented_count < needed:
            # Cycle through images
            img_path = image_files[aug_cycle % len(image_files)]
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                aug_cycle += 1
                continue
            
            # Choose random augmentation
            aug_type = random.choice(augment_types)
            
            # Apply augmentation
            aug_img = apply_geometric_augmentation(img, aug_type)
            
            # Save augmented image
            aug_filename = f"{img_path.stem}_aug{augmented_count}_{aug_type}{img_path.suffix}"
            aug_path = minority_dir / aug_filename
            
            cv2.imwrite(str(aug_path), aug_img)
            
            augmented_count += 1
            pbar.update(1)
            aug_cycle += 1
    
    # Final count
    final_count = count_images_in_folder(minority_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ DENGELEME TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  • {minority_class}: {minority_count:,} → {final_count:,} (+{augmented_count:,})")
    print(f"  • Yeni oran: {max(majority_count, final_count) / min(majority_count, final_count):.2f}:1")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    dataset_path = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    print("⚖️  Crop Dataset Class Balancer")
    print("=" * 60)
    print("Bu script crop'lanmış dataset'i geometric augmentation")
    print("ile dengeler. Color-based model için color augmentation")
    print("YAPILMAZ (sadece flip, rotate, perspective).")
    print("=" * 60)
    
    balance_ratio = float(input("\nDenge oranı (0.0-1.0, örn: 1.0 tam denge): "))
    
    balance_cropped_dataset(
        dataset_path=dataset_path,
        balance_ratio=balance_ratio,
        augment_types=['flip', 'rotate', 'perspective']
    )
