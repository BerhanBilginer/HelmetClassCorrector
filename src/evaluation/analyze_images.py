import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import os

class ImageAnalyzer:
    def __init__(self, helmet_dir, no_helmet_dir):
        self.helmet_dir = Path(helmet_dir)
        self.no_helmet_dir = Path(no_helmet_dir)
        self.results = []
        
    def analyze_image(self, image_path, category):
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        height, width = img.shape[:2]
        
        mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))
        std_r, std_g, std_b = img_rgb.std(axis=(0, 1))
        
        brightness = img_gray.mean()
        contrast = img_gray.std()
        
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = (edges > 0).sum() / (height * width)
        
        hue_mean = img_hsv[:, :, 0].mean()
        saturation_mean = img_hsv[:, :, 1].mean()
        value_mean = img_hsv[:, :, 2].mean()
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        file_size = os.path.getsize(image_path)
        
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        
        entropy_r = -np.sum((hist_r / hist_r.sum()) * np.log2((hist_r / hist_r.sum()) + 1e-10))
        entropy_g = -np.sum((hist_g / hist_g.sum()) * np.log2((hist_g / hist_g.sum()) + 1e-10))
        entropy_b = -np.sum((hist_b / hist_b.sum()) * np.log2((hist_b / hist_b.sum()) + 1e-10))
        
        return {
            'filename': image_path.name,
            'category': category,
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'total_pixels': width * height,
            'file_size_kb': file_size / 1024,
            'mean_red': mean_r,
            'mean_green': mean_g,
            'mean_blue': mean_b,
            'std_red': std_r,
            'std_green': std_g,
            'std_blue': std_b,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'hue_mean': hue_mean,
            'saturation_mean': saturation_mean,
            'value_mean': value_mean,
            'sharpness': sharpness,
            'entropy_red': entropy_r,
            'entropy_green': entropy_g,
            'entropy_blue': entropy_b,
        }
    
    def analyze_all_images(self):
        for img_path in sorted(self.helmet_dir.glob('*.png')):
            result = self.analyze_image(img_path, 'helmet')
            if result:
                self.results.append(result)
        
        for img_path in sorted(self.no_helmet_dir.glob('*.png')):
            result = self.analyze_image(img_path, 'no_helmet')
            if result:
                self.results.append(result)
        
        self.df = pd.DataFrame(self.results)
        return self.df
    
    def create_visualization(self, save_path='image_analysis_report.png'):
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(8, 3, hspace=0.4, wspace=0.3)
        
        helmet_data = self.df[self.df['category'] == 'helmet']
        no_helmet_data = self.df[self.df['category'] == 'no_helmet']
        
        colors = {'helmet': '#2ecc71', 'no_helmet': '#e74c3c'}
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.5, 0.5, 'HELMET vs NO_HELMET - Detaylı Görüntü Analizi', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax1.text(0.5, 0.2, f'Toplam Görüntü: {len(self.df)} (Helmet: {len(helmet_data)}, No Helmet: {len(no_helmet_data)})', 
                ha='center', va='center', fontsize=14)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1, 0])
        brightness_data = [helmet_data['brightness'].values, no_helmet_data['brightness'].values]
        bp = ax2.boxplot(brightness_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax2.set_ylabel('Parlaklık (Brightness)')
        ax2.set_title('Parlaklık Karşılaştırması')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        contrast_data = [helmet_data['contrast'].values, no_helmet_data['contrast'].values]
        bp = ax3.boxplot(contrast_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax3.set_ylabel('Kontrast')
        ax3.set_title('Kontrast Karşılaştırması')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        sharpness_data = [helmet_data['sharpness'].values, no_helmet_data['sharpness'].values]
        bp = ax4.boxplot(sharpness_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax4.set_ylabel('Keskinlik (Sharpness)')
        ax4.set_title('Keskinlik Karşılaştırması')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(helmet_data['mean_red'], bins=15, alpha=0.6, label='Helmet', color=colors['helmet'])
        ax5.hist(no_helmet_data['mean_red'], bins=15, alpha=0.6, label='No Helmet', color=colors['no_helmet'])
        ax5.set_xlabel('Ortalama Kırmızı Değeri')
        ax5.set_ylabel('Frekans')
        ax5.set_title('Kırmızı Kanal Dağılımı')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(helmet_data['mean_green'], bins=15, alpha=0.6, label='Helmet', color=colors['helmet'])
        ax6.hist(no_helmet_data['mean_green'], bins=15, alpha=0.6, label='No Helmet', color=colors['no_helmet'])
        ax6.set_xlabel('Ortalama Yeşil Değeri')
        ax6.set_ylabel('Frekans')
        ax6.set_title('Yeşil Kanal Dağılımı')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(helmet_data['mean_blue'], bins=15, alpha=0.6, label='Helmet', color=colors['helmet'])
        ax7.hist(no_helmet_data['mean_blue'], bins=15, alpha=0.6, label='No Helmet', color=colors['no_helmet'])
        ax7.set_xlabel('Ortalama Mavi Değeri')
        ax7.set_ylabel('Frekans')
        ax7.set_title('Mavi Kanal Dağılımı')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[3, 0])
        edge_data = [helmet_data['edge_density'].values, no_helmet_data['edge_density'].values]
        bp = ax8.boxplot(edge_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax8.set_ylabel('Kenar Yoğunluğu')
        ax8.set_title('Kenar Yoğunluğu Karşılaştırması')
        ax8.grid(True, alpha=0.3)
        
        ax9 = fig.add_subplot(gs[3, 1])
        saturation_data = [helmet_data['saturation_mean'].values, no_helmet_data['saturation_mean'].values]
        bp = ax9.boxplot(saturation_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax9.set_ylabel('Doygunluk (Saturation)')
        ax9.set_title('Doygunluk Karşılaştırması')
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(gs[3, 2])
        hue_data = [helmet_data['hue_mean'].values, no_helmet_data['hue_mean'].values]
        bp = ax10.boxplot(hue_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax10.set_ylabel('Renk Tonu (Hue)')
        ax10.set_title('Renk Tonu Karşılaştırması')
        ax10.grid(True, alpha=0.3)
        
        ax11 = fig.add_subplot(gs[4, 0])
        ax11.scatter(helmet_data['width'], helmet_data['height'], 
                    alpha=0.6, s=100, label='Helmet', color=colors['helmet'])
        ax11.scatter(no_helmet_data['width'], no_helmet_data['height'], 
                    alpha=0.6, s=100, label='No Helmet', color=colors['no_helmet'])
        ax11.set_xlabel('Genişlik (Width)')
        ax11.set_ylabel('Yükseklik (Height)')
        ax11.set_title('Görüntü Boyutları')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        ax12 = fig.add_subplot(gs[4, 1])
        ax12.hist(helmet_data['aspect_ratio'], bins=10, alpha=0.6, label='Helmet', color=colors['helmet'])
        ax12.hist(no_helmet_data['aspect_ratio'], bins=10, alpha=0.6, label='No Helmet', color=colors['no_helmet'])
        ax12.set_xlabel('En-Boy Oranı')
        ax12.set_ylabel('Frekans')
        ax12.set_title('En-Boy Oranı Dağılımı')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        
        ax13 = fig.add_subplot(gs[4, 2])
        ax13.bar(['Helmet', 'No Helmet'], 
                [helmet_data['file_size_kb'].mean(), no_helmet_data['file_size_kb'].mean()],
                color=[colors['helmet'], colors['no_helmet']], alpha=0.7)
        ax13.set_ylabel('Ortalama Dosya Boyutu (KB)')
        ax13.set_title('Ortalama Dosya Boyutu')
        ax13.grid(True, alpha=0.3, axis='y')
        
        ax14 = fig.add_subplot(gs[5, :])
        metrics = ['brightness', 'contrast', 'sharpness', 'edge_density', 'saturation_mean']
        helmet_means = [helmet_data[m].mean() for m in metrics]
        no_helmet_means = [no_helmet_data[m].mean() for m in metrics]
        
        helmet_normalized = [v / max(helmet_means[i], no_helmet_means[i]) for i, v in enumerate(helmet_means)]
        no_helmet_normalized = [v / max(helmet_means[i], no_helmet_means[i]) for i, v in enumerate(no_helmet_means)]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax14.bar(x - width/2, helmet_normalized, width, label='Helmet', color=colors['helmet'], alpha=0.7)
        ax14.bar(x + width/2, no_helmet_normalized, width, label='No Helmet', color=colors['no_helmet'], alpha=0.7)
        ax14.set_ylabel('Normalize Edilmiş Değer')
        ax14.set_title('Ana Metriklerin Karşılaştırması (Normalize)')
        ax14.set_xticks(x)
        ax14.set_xticklabels(['Parlaklık', 'Kontrast', 'Keskinlik', 'Kenar Yoğ.', 'Doygunluk'])
        ax14.legend()
        ax14.grid(True, alpha=0.3, axis='y')
        
        ax15 = fig.add_subplot(gs[6, 0])
        entropy_data = [helmet_data['entropy_red'].values, no_helmet_data['entropy_red'].values]
        bp = ax15.boxplot(entropy_data, labels=['Helmet', 'No Helmet'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['helmet'])
        bp['boxes'][1].set_facecolor(colors['no_helmet'])
        ax15.set_ylabel('Entropi (Kırmızı)')
        ax15.set_title('Kırmızı Kanal Entropi')
        ax15.grid(True, alpha=0.3)
        
        ax16 = fig.add_subplot(gs[6, 1])
        ax16.scatter(helmet_data['brightness'], helmet_data['contrast'], 
                    alpha=0.6, s=100, label='Helmet', color=colors['helmet'])
        ax16.scatter(no_helmet_data['brightness'], no_helmet_data['contrast'], 
                    alpha=0.6, s=100, label='No Helmet', color=colors['no_helmet'])
        ax16.set_xlabel('Parlaklık')
        ax16.set_ylabel('Kontrast')
        ax16.set_title('Parlaklık vs Kontrast')
        ax16.legend()
        ax16.grid(True, alpha=0.3)
        
        ax17 = fig.add_subplot(gs[6, 2])
        ax17.scatter(helmet_data['saturation_mean'], helmet_data['value_mean'], 
                    alpha=0.6, s=100, label='Helmet', color=colors['helmet'])
        ax17.scatter(no_helmet_data['saturation_mean'], no_helmet_data['value_mean'], 
                    alpha=0.6, s=100, label='No Helmet', color=colors['no_helmet'])
        ax17.set_xlabel('Doygunluk')
        ax17.set_ylabel('Değer (Value)')
        ax17.set_title('Doygunluk vs Değer')
        ax17.legend()
        ax17.grid(True, alpha=0.3)
        
        ax18 = fig.add_subplot(gs[7, :])
        stats_text = "İSTATİSTİKSEL ÖZET\n\n"
        
        stats_text += "HELMET KLASÖRÜ:\n"
        stats_text += f"  • Görüntü Sayısı: {len(helmet_data)}\n"
        stats_text += f"  • Ort. Parlaklık: {helmet_data['brightness'].mean():.2f} ± {helmet_data['brightness'].std():.2f}\n"
        stats_text += f"  • Ort. Kontrast: {helmet_data['contrast'].mean():.2f} ± {helmet_data['contrast'].std():.2f}\n"
        stats_text += f"  • Ort. Keskinlik: {helmet_data['sharpness'].mean():.2f} ± {helmet_data['sharpness'].std():.2f}\n"
        stats_text += f"  • Ort. Kenar Yoğunluğu: {helmet_data['edge_density'].mean():.4f} ± {helmet_data['edge_density'].std():.4f}\n"
        stats_text += f"  • Ort. Dosya Boyutu: {helmet_data['file_size_kb'].mean():.2f} KB\n\n"
        
        stats_text += "NO_HELMET KLASÖRÜ:\n"
        stats_text += f"  • Görüntü Sayısı: {len(no_helmet_data)}\n"
        stats_text += f"  • Ort. Parlaklık: {no_helmet_data['brightness'].mean():.2f} ± {no_helmet_data['brightness'].std():.2f}\n"
        stats_text += f"  • Ort. Kontrast: {no_helmet_data['contrast'].mean():.2f} ± {no_helmet_data['contrast'].std():.2f}\n"
        stats_text += f"  • Ort. Keskinlik: {no_helmet_data['sharpness'].mean():.2f} ± {no_helmet_data['sharpness'].std():.2f}\n"
        stats_text += f"  • Ort. Kenar Yoğunluğu: {no_helmet_data['edge_density'].mean():.4f} ± {no_helmet_data['edge_density'].std():.4f}\n"
        stats_text += f"  • Ort. Dosya Boyutu: {no_helmet_data['file_size_kb'].mean():.2f} KB\n\n"
        
        stats_text += "FARKLAR:\n"
        stats_text += f"  • Parlaklık Farkı: {abs(helmet_data['brightness'].mean() - no_helmet_data['brightness'].mean()):.2f}\n"
        stats_text += f"  • Kontrast Farkı: {abs(helmet_data['contrast'].mean() - no_helmet_data['contrast'].mean()):.2f}\n"
        stats_text += f"  • Keskinlik Farkı: {abs(helmet_data['sharpness'].mean() - no_helmet_data['sharpness'].mean()):.2f}\n"
        stats_text += f"  • Kenar Yoğunluğu Farkı: {abs(helmet_data['edge_density'].mean() - no_helmet_data['edge_density'].mean()):.4f}\n"
        
        ax18.text(0.05, 0.95, stats_text, transform=ax18.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax18.axis('off')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Görsel rapor kaydedildi: {save_path}")
        plt.show()
        
    def save_to_csv(self, filename='image_analysis.csv'):
        self.df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ CSV dosyası kaydedildi: {filename}")
        
    def save_to_excel(self, filename='image_analysis.xlsx'):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Tüm Veriler', index=False)
            
            helmet_data = self.df[self.df['category'] == 'helmet']
            no_helmet_data = self.df[self.df['category'] == 'no_helmet']
            
            helmet_data.to_excel(writer, sheet_name='Helmet', index=False)
            no_helmet_data.to_excel(writer, sheet_name='No Helmet', index=False)
            
            summary_data = {
                'Metrik': ['Görüntü Sayısı', 'Ort. Parlaklık', 'Ort. Kontrast', 
                          'Ort. Keskinlik', 'Ort. Kenar Yoğunluğu', 'Ort. Doygunluk',
                          'Ort. Dosya Boyutu (KB)'],
                'Helmet': [
                    len(helmet_data),
                    helmet_data['brightness'].mean(),
                    helmet_data['contrast'].mean(),
                    helmet_data['sharpness'].mean(),
                    helmet_data['edge_density'].mean(),
                    helmet_data['saturation_mean'].mean(),
                    helmet_data['file_size_kb'].mean()
                ],
                'No Helmet': [
                    len(no_helmet_data),
                    no_helmet_data['brightness'].mean(),
                    no_helmet_data['contrast'].mean(),
                    no_helmet_data['s
2026-02-08 12:55:20.491 Serialization of dataframe to Arrow table was unsuccessful. Applying automatic fixes for column types to make the dataframe Arrow-compatible.
Traceback (most recent call last):
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/streamlit/dataframe_util.py", line 829, in convert_pandas_df_to_arrow_bytes
    table = pa.Table.from_pandas(df)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/table.pxi", line 4795, in pyarrow.lib.Table.from_pandas
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 637, in dataframe_to_arrays
    arrays = [convert_column(c, f)
              ^^^^^^^^^^^^^^^^^^^^
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 625, in convert_column
    raise e
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 619, in convert_column
    result = pa.array(col, type=type_, from_pandas=True, safe=safe)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/array.pxi", line 365, in pyarrow.lib.array
  File "pyarrow/array.pxi", line 91, in pyarrow.lib._ndarray_to_array
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: ("Could not convert '114.47 ± 37.31' with type str: tried to convert to int64", 'Conversion failed for column Değer with type object')
2026-02-08 12:55:20.496 Serialization of dataframe to Arrow table was unsuccessful. Applying automatic fixes for column types to make the dataframe Arrow-compatible.
Traceback (most recent call last):
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/streamlit/dataframe_util.py", line 829, in convert_pandas_df_to_arrow_bytes
    table = pa.Table.from_pandas(df)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/table.pxi", line 4795, in pyarrow.lib.Table.from_pandas
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 637, in dataframe_to_arrays
    arrays = [convert_column(c, f)
              ^^^^^^^^^^^^^^^^^^^^
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 625, in convert_column
    raise e
  File "/home/berhan/.pyenv/versions/3.12.3/envs/big-eye-env/lib/python3.12/site-packages/pyarrow/pandas_compat.py", line 619, in convert_column
    result = pa.array(col, type=type_, from_pandas=True, safe=safe)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/array.pxi", line 365, in pyarrow.lib.array
  File "pyarrow/array.pxi", line 91, in pyarrow.lib._ndarray_to_array
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: ("Could not convert '88.73 ± 29.65' with type str: tried to convert to int64", 'Conversion failed for column Değer with type object')
harpness'].mean(),
                    no_helmet_data['edge_density'].mean(),
                    no_helmet_data['saturation_mean'].mean(),
                    no_helmet_data['file_size_kb'].mean()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df['Fark'] = abs(summary_df['Helmet'] - summary_df['No Helmet'])
            summary_df.to_excel(writer, sheet_name='Özet', index=False)
            
        print(f"✓ Excel dosyası kaydedildi: {filename}")

def main():
    helmet_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images/helmet'
    no_helmet_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images/no_helmet'
    
    print("=" * 60)
    print("HELMET vs NO_HELMET - Görüntü Analizi")
    print("=" * 60)
    
    analyzer = ImageAnalyzer(helmet_dir, no_helmet_dir)
    
    print("\n[1/4] Görüntüler analiz ediliyor...")
    df = analyzer.analyze_all_images()
    print(f"✓ {len(df)} görüntü analiz edildi")
    
    print("\n[2/4] Görsel rapor oluşturuluyor...")
    analyzer.create_visualization('image_analysis_report.png')
    
    print("\n[3/4] CSV dosyası kaydediliyor...")
    analyzer.save_to_csv('image_analysis.csv')
    
    print("\n[4/4] Excel dosyası kaydediliyor...")
    analyzer.save_to_excel('image_analysis.xlsx')
    
    print("\n" + "=" * 60)
    print("✓ Analiz tamamlandı!")
    print("=" * 60)
    print("\nOluşturulan dosyalar:")
    print("  • image_analysis_report.png - Detaylı görsel rapor")
    print("  • image_analysis.csv - CSV formatında tüm veriler")
    print("  • image_analysis.xlsx - Excel formatında tüm veriler (çoklu sayfa)")
    print("=" * 60)

if __name__ == "__main__":
    main()
