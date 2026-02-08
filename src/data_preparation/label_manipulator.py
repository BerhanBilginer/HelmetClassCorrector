import os
from pathlib import Path
from tqdm import tqdm

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
    
    print("🔧 YOLO Label Filtreleme Aracı")
    print("=" * 60)
    print("Bu script class_id 0 ve 2 olan satırları siler,")
    print("sadece class_id 1 ve 3 olanları tutar.")
    print("=" * 60)
    
    response = input("\nDevam etmek istiyor musunuz? (y/n): ").lower()
    
    if response == 'y':
        stats = filter_labels_by_class(
            dataset_path=dataset_path,
            keep_classes=[1, 3],
            backup=True
        )
        
        print("\n💡 İpucu: Geri yüklemek için restore_from_backup() fonksiyonunu kullanabilirsiniz.")
    else:
        print("❌ İşlem iptal edildi.")
