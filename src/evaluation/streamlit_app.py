import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
import os

st.set_page_config(page_title="Helmet vs No-Helmet Analizi", layout="wide", page_icon="🪖")

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
            'filename': image_path.name, 'category': category, 'width': width, 'height': height,
            'aspect_ratio': width / height, 'total_pixels': width * height, 'file_size_kb': file_size / 1024,
            'mean_red': mean_r, 'mean_green': mean_g, 'mean_blue': mean_b,
            'std_red': std_r, 'std_green': std_g, 'std_blue': std_b,
            'brightness': brightness, 'contrast': contrast, 'edge_density': edge_density,
            'hue_mean': hue_mean, 'saturation_mean': saturation_mean, 'value_mean': value_mean,
            'sharpness': sharpness, 'entropy_red': entropy_r, 'entropy_green': entropy_g, 'entropy_blue': entropy_b,
        }
    
    def analyze_all_images(self):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_images = list(self.helmet_dir.glob('*.png')) + list(self.no_helmet_dir.glob('*.png'))
        total = len(all_images)
        
        for idx, img_path in enumerate(sorted(self.helmet_dir.glob('*.png'))):
            status_text.text(f'Analiz ediliyor: {img_path.name}')
            result = self.analyze_image(img_path, 'helmet')
            if result:
                self.results.append(result)
            progress_bar.progress((idx + 1) / total)
        
        helmet_count = len(self.results)
        for idx, img_path in enumerate(sorted(self.no_helmet_dir.glob('*.png'))):
            status_text.text(f'Analiz ediliyor: {img_path.name}')
            result = self.analyze_image(img_path, 'no_helmet')
            if result:
                self.results.append(result)
            progress_bar.progress((helmet_count + idx + 1) / total)
        
        progress_bar.empty()
        status_text.empty()
        self.df = pd.DataFrame(self.results)
        return self.df
    
    def save_to_excel(self, filename='image_analysis.xlsx'):
        helmet_data = self.df[self.df['category'] == 'helmet']
        no_helmet_data = self.df[self.df['category'] == 'no_helmet']
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Tüm Veriler', index=False)
            helmet_data.to_excel(writer, sheet_name='Helmet', index=False)
            no_helmet_data.to_excel(writer, sheet_name='No Helmet', index=False)
            
            summary_df = pd.DataFrame({
                'Metrik': ['Görüntü Sayısı', 'Ort. Parlaklık', 'Ort. Kontrast', 'Ort. Keskinlik', 
                          'Ort. Kenar Yoğunluğu', 'Ort. Doygunluk', 'Ort. Dosya Boyutu (KB)'],
                'Helmet': [len(helmet_data), helmet_data['brightness'].mean(), helmet_data['contrast'].mean(),
                          helmet_data['sharpness'].mean(), helmet_data['edge_density'].mean(),
                          helmet_data['saturation_mean'].mean(), helmet_data['file_size_kb'].mean()],
                'No Helmet': [len(no_helmet_data), no_helmet_data['brightness'].mean(), no_helmet_data['contrast'].mean(),
                             no_helmet_data['sharpness'].mean(), no_helmet_data['edge_density'].mean(),
                             no_helmet_data['saturation_mean'].mean(), no_helmet_data['file_size_kb'].mean()]
            })
            summary_df['Fark'] = abs(summary_df['Helmet'] - summary_df['No Helmet'])
            summary_df.to_excel(writer, sheet_name='Özet', index=False)
        return filename

@st.cache_data
def load_and_analyze_data():
    helmet_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images/helmet'
    no_helmet_dir = '/home/berhan/Development/personal/HelmetClassCorrector/test/cropped_images/no_helmet'
    analyzer = ImageAnalyzer(helmet_dir, no_helmet_dir)
    df = analyzer.analyze_all_images()
    return df, analyzer

def main():
    st.title("🪖 Helmet vs No-Helmet Görüntü Analizi")
    st.markdown("---")
    
    with st.spinner('Görüntüler analiz ediliyor...'):
        df, analyzer = load_and_analyze_data()
    
    helmet_data = df[df['category'] == 'helmet']
    no_helmet_data = df[df['category'] == 'no_helmet']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Görüntü", len(df))
    with col2:
        st.metric("Helmet", len(helmet_data))
    with col3:
        st.metric("No Helmet", len(no_helmet_data))
    with col4:
        st.metric("Oran", f"{len(helmet_data)}/{len(no_helmet_data)}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Karşılaştırma", "🎨 Renk Analizi", "📈 İstatistikler", "💾 Veri İndirme"])
    
    with tab1:
        st.header("Genel Karşılaştırma")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Parlaklık', 'Kontrast', 'Keskinlik', 'Kenar Yoğunluğu'),
            vertical_spacing=0.15, horizontal_spacing=0.12
        )
        
        metrics = [('brightness', 1, 1), ('contrast', 1, 2), ('sharpness', 2, 1), ('edge_density', 2, 2)]
        
        for metric, row, col in metrics:
            fig.add_trace(go.Box(y=helmet_data[metric], name='Helmet', marker_color='#2ecc71', 
                                showlegend=(row==1 and col==1)), row=row, col=col)
            fig.add_trace(go.Box(y=no_helmet_data[metric], name='No Helmet', marker_color='#e74c3c', 
                                showlegend=(row==1 and col==1)), row=row, col=col)
        
        fig.update_layout(height=600, title_text="Temel Metrikler - Box Plot Karşılaştırması", showlegend=True)
        st.plotly_chart(fig, width='stretch')
        
        st.subheader("Normalize Edilmiş Metrik Karşılaştırması")
        comparison_metrics = ['brightness', 'contrast', 'sharpness', 'edge_density', 'saturation_mean']
        helmet_means = [helmet_data[m].mean() for m in comparison_metrics]
        no_helmet_means = [no_helmet_data[m].mean() for m in comparison_metrics]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=['Parlaklık', 'Kontrast', 'Keskinlik', 'Kenar Yoğ.', 'Doygunluk'],
                             y=helmet_means, name='Helmet', marker_color='#2ecc71'))
        fig2.add_trace(go.Bar(x=['Parlaklık', 'Kontrast', 'Keskinlik', 'Kenar Yoğ.', 'Doygunluk'],
                             y=no_helmet_means, name='No Helmet', marker_color='#e74c3c'))
        fig2.update_layout(barmode='group', height=400, title="Ortalama Değerler Karşılaştırması")
        st.plotly_chart(fig2, width='stretch')
    
    with tab2:
        st.header("Renk Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=helmet_data['mean_red'], name='Helmet - Kırmızı', 
                                       marker_color='rgba(255, 0, 0, 0.5)', nbinsx=20))
            fig3.add_trace(go.Histogram(x=no_helmet_data['mean_red'], name='No Helmet - Kırmızı', 
                                       marker_color='rgba(255, 0, 0, 0.3)', nbinsx=20))
            fig3.update_layout(barmode='overlay', title="Kırmızı Kanal Dağılımı", height=350)
            st.plotly_chart(fig3, width='stretch')
            
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(x=helmet_data['mean_green'], name='Helmet - Yeşil', 
                                       marker_color='rgba(0, 255, 0, 0.5)', nbinsx=20))
            fig4.add_trace(go.Histogram(x=no_helmet_data['mean_green'], name='No Helmet - Yeşil', 
                                       marker_color='rgba(0, 255, 0, 0.3)', nbinsx=20))
            fig4.update_layout(barmode='overlay', title="Yeşil Kanal Dağılımı", height=350)
            st.plotly_chart(fig4, width='stretch')
        
        with col2:
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(x=helmet_data['mean_blue'], name='Helmet - Mavi', 
                                       marker_color='rgba(0, 0, 255, 0.5)', nbinsx=20))
            fig5.add_trace(go.Histogram(x=no_helmet_data['mean_blue'], name='No Helmet - Mavi', 
                                       marker_color='rgba(0, 0, 255, 0.3)', nbinsx=20))
            fig5.update_layout(barmode='overlay', title="Mavi Kanal Dağılımı", height=350)
            st.plotly_chart(fig5, width='stretch')
            
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=helmet_data['saturation_mean'], y=helmet_data['value_mean'],
                                     mode='markers', name='Helmet', marker=dict(size=10, color='#2ecc71')))
            fig6.add_trace(go.Scatter(x=no_helmet_data['saturation_mean'], y=no_helmet_data['value_mean'],
                                     mode='markers', name='No Helmet', marker=dict(size=10, color='#e74c3c')))
            fig6.update_layout(title="Doygunluk vs Değer", xaxis_title="Doygunluk", yaxis_title="Değer", height=350)
            st.plotly_chart(fig6, width='stretch')
    
    with tab3:
        st.header("İstatistiksel Özet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Helmet Klasörü")
            stats_helmet = pd.DataFrame({
                'Metrik': ['Görüntü Sayısı', 'Ort. Parlaklık', 'Ort. Kontrast', 'Ort. Keskinlik', 
                          'Ort. Kenar Yoğunluğu', 'Ort. Doygunluk', 'Ort. Dosya Boyutu (KB)'],
                'Değer': [
                    str(len(helmet_data)),
                    f"{helmet_data['brightness'].mean():.2f} ± {helmet_data['brightness'].std():.2f}",
                    f"{helmet_data['contrast'].mean():.2f} ± {helmet_data['contrast'].std():.2f}",
                    f"{helmet_data['sharpness'].mean():.2f} ± {helmet_data['sharpness'].std():.2f}",
                    f"{helmet_data['edge_density'].mean():.4f} ± {helmet_data['edge_density'].std():.4f}",
                    f"{helmet_data['saturation_mean'].mean():.2f} ± {helmet_data['saturation_mean'].std():.2f}",
                    f"{helmet_data['file_size_kb'].mean():.2f}"
                ]
            })
            stats_helmet['Değer'] = stats_helmet['Değer'].astype(str)
            st.dataframe(stats_helmet, width='stretch', hide_index=True)
        
        with col2:
            st.subheader("🔴 No Helmet Klasörü")
            stats_no_helmet = pd.DataFrame({
                'Metrik': ['Görüntü Sayısı', 'Ort. Parlaklık', 'Ort. Kontrast', 'Ort. Keskinlik', 
                          'Ort. Kenar Yoğunluğu', 'Ort. Doygunluk', 'Ort. Dosya Boyutu (KB)'],
                'Değer': [
                    str(len(no_helmet_data)),
                    f"{no_helmet_data['brightness'].mean():.2f} ± {no_helmet_data['brightness'].std():.2f}",
                    f"{no_helmet_data['contrast'].mean():.2f} ± {no_helmet_data['contrast'].std():.2f}",
                    f"{no_helmet_data['sharpness'].mean():.2f} ± {no_helmet_data['sharpness'].std():.2f}",
                    f"{no_helmet_data['edge_density'].mean():.4f} ± {no_helmet_data['edge_density'].std():.4f}",
                    f"{no_helmet_data['saturation_mean'].mean():.2f} ± {no_helmet_data['saturation_mean'].std():.2f}",
                    f"{no_helmet_data['file_size_kb'].mean():.2f}"
                ]
            })
            stats_no_helmet['Değer'] = stats_no_helmet['Değer'].astype(str)
            st.dataframe(stats_no_helmet, width='stretch', hide_index=True)
        
        st.subheader("📊 Farklar")
        differences = pd.DataFrame({
            'Metrik': ['Parlaklık', 'Kontrast', 'Keskinlik', 'Kenar Yoğunluğu', 'Doygunluk'],
            'Fark': [
                abs(helmet_data['brightness'].mean() - no_helmet_data['brightness'].mean()),
                abs(helmet_data['contrast'].mean() - no_helmet_data['contrast'].mean()),
                abs(helmet_data['sharpness'].mean() - no_helmet_data['sharpness'].mean()),
                abs(helmet_data['edge_density'].mean() - no_helmet_data['edge_density'].mean()),
                abs(helmet_data['saturation_mean'].mean() - no_helmet_data['saturation_mean'].mean())
            ]
        })
        st.dataframe(differences, width='stretch', hide_index=True)
        
        st.subheader("📋 Tüm Veriler")
        st.dataframe(df, width='stretch', height=400)
    
    with tab4:
        st.header("Veri İndirme")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV İndir",
                data=csv_data,
                file_name="image_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("📥 Excel Oluştur ve İndir"):
                excel_file = analyzer.save_to_excel('image_analysis.xlsx')
                with open(excel_file, 'rb') as f:
                    st.download_button(
                        label="📥 Excel Dosyasını İndir",
                        data=f,
                        file_name="image_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                st.success("✅ Excel dosyası oluşturuldu!")

if __name__ == "__main__":
    main()
