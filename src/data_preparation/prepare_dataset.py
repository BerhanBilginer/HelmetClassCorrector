import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def prepare_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Cropped görüntüleri train/val/test olarak böler.
    
    Args:
        source_dir: Cropped images klasörü (helmet ve no_helmet içeren)
        output_dir: Çıktı klasörü
        train_ratio: Training oranı
        val_ratio: Validation oranı
        test_ratio: Test oranı
        seed: Random seed
    """
    random.seed(seed)
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    classes = ['helmet', 'no_helmet']
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'helmet': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'no_helmet': {'train': 0, 'val': 0, 'test': 0, 'total': 0}
    }
    
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Split: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    print("=" * 60)
    
    for cls in classes:
        cls_dir = source_dir / cls
        
        if not cls_dir.exists():
            print(f"⚠️  {cls} klasörü bulunamadı!")
            continue
        
        images = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpg'))
        
        if not images:
            print(f"⚠️  {cls} klasöründe görüntü bulunamadı!")
            continue
        
        stats[cls]['total'] = len(images)
        
        train_images, temp_images = train_test_split(
            images, 
            test_size=(val_ratio + test_ratio),
            random_state=seed
        )
        
        val_images, test_images = train_test_split(
            temp_images,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed
        )
        
        print(f"\n📁 {cls.upper()}")
        print(f"  Total: {len(images)}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val: {len(val_images)}")
        print(f"  Test: {len(test_images)}")
        
        for img in train_images:
            dst = output_dir / 'train' / cls / img.name
            shutil.copy2(img, dst)
            stats[cls]['train'] += 1
        
        for img in val_images:
            dst = output_dir / 'val' / cls / img.name
            shutil.copy2(img, dst)
            stats[cls]['val'] += 1
        
        for img in test_images:
            dst = output_dir / 'test' / cls / img.name
            shutil.copy2(img, dst)
            stats[cls]['test'] += 1
    
    print("\n" + "=" * 60)
    print("✅ DATASET HAZIR")
    print("=" * 60)
    print("\n📊 ÖZET:")
    print(f"\n{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    for cls in classes:
        print(f"{cls:<15} {stats[cls]['train']:<10} {stats[cls]['val']:<10} {stats[cls]['test']:<10} {stats[cls]['total']:<10}")
    
    total_train = sum(stats[cls]['train'] for cls in classes)
    total_val = sum(stats[cls]['val'] for cls in classes)
    total_test = sum(stats[cls]['test'] for cls in classes)
    total_all = sum(stats[cls]['total'] for cls in classes)
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    print("=" * 60)
    
    return stats

if __name__ == "__main__":
    source_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images'
    output_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    print("🔧 Dataset Preparation Tool")
    print("=" * 60)
    print("Bu script cropped görüntüleri train/val/test olarak böler.")
    print("=" * 60)
    
    response = input("\nDevam etmek istiyor musunuz? (y/n): ").lower()
    
    if response == 'y':
        stats = prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        print("\n✅ İşlem tamamlandı!")
    else:
        print("❌ İşlem iptal edildi.")
