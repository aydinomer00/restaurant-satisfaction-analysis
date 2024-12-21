import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Giriş ve çıkış değişkenlerini tanımlama
service_speed = ctrl.Antecedent(np.arange(0, 11, 1), 'service_speed')
food_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'food_quality')
customer_satisfaction = ctrl.Consequent(np.arange(0, 11, 1), 'customer_satisfaction')

# Üyelik fonksiyonlarını tanımlama
service_speed['slow'] = fuzz.trimf(service_speed.universe, [0, 0, 5])
service_speed['normal'] = fuzz.trimf(service_speed.universe, [2, 5, 8])
service_speed['fast'] = fuzz.trimf(service_speed.universe, [5, 10, 10])

food_quality['bad'] = fuzz.trimf(food_quality.universe, [0, 0, 5])
food_quality['good'] = fuzz.trimf(food_quality.universe, [2, 5, 8])
food_quality['excellent'] = fuzz.trimf(food_quality.universe, [5, 10, 10])

customer_satisfaction['low'] = fuzz.trimf(customer_satisfaction.universe, [0, 0, 5])
customer_satisfaction['medium'] = fuzz.trimf(customer_satisfaction.universe, [2, 5, 8])
customer_satisfaction['high'] = fuzz.trimf(customer_satisfaction.universe, [5, 10, 10])

# Kural tabanını oluşturma
rule1 = ctrl.Rule(service_speed['fast'] & food_quality['excellent'], customer_satisfaction['high'])
rule2 = ctrl.Rule(service_speed['slow'] & food_quality['bad'], customer_satisfaction['low'])
rule3 = ctrl.Rule(service_speed['normal'] & food_quality['good'], customer_satisfaction['medium'])
rule4 = ctrl.Rule(service_speed['fast'] & food_quality['bad'], customer_satisfaction['medium'])
rule5 = ctrl.Rule(service_speed['slow'] & food_quality['excellent'], customer_satisfaction['medium'])
rule6 = ctrl.Rule(service_speed['normal'] & food_quality['excellent'], customer_satisfaction['high'])
rule7 = ctrl.Rule(service_speed['normal'] & food_quality['bad'], customer_satisfaction['low'])
rule8 = ctrl.Rule(service_speed['fast'] & food_quality['good'], customer_satisfaction['high'])
rule9 = ctrl.Rule(service_speed['slow'] & food_quality['good'], customer_satisfaction['medium'])

# Kontrol sistemini oluşturma
satisfaction_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
satisfaction_simulator = ctrl.ControlSystemSimulation(satisfaction_ctrl)


def generate_and_analyze_data(n_samples=500):
    """Sentetik veri üretir ve analiz eder"""
    np.random.seed(42)
    service_speeds = np.random.uniform(0, 10, n_samples)
    food_qualities = np.random.uniform(0, 10, n_samples)

    results = []
    for speed, quality in zip(service_speeds, food_qualities):
        satisfaction_simulator.input['service_speed'] = speed
        satisfaction_simulator.input['food_quality'] = quality
        satisfaction_simulator.compute()
        satisfaction = satisfaction_simulator.output['customer_satisfaction']
        results.append([speed, quality, satisfaction])

    df = pd.DataFrame(results, columns=['Hizmet_Hizi', 'Yemek_Kalitesi', 'Musteri_Memnuniyeti'])
    return df


def create_visualizations(df):
    """Tüm görselleştirmeleri tek bir fonksiyonda toplar"""

    # 1. Temel Analiz Görselleştirmesi
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(df['Hizmet_Hizi'], df['Yemek_Kalitesi'], c=df['Musteri_Memnuniyeti'], cmap='viridis')
    plt.colorbar(label='Müşteri Memnuniyeti')
    plt.xlabel('Hizmet Hızı')
    plt.ylabel('Yemek Kalitesi')
    plt.title('Müşteri Memnuniyeti Dağılımı')

    plt.subplot(2, 2, 2)
    plt.hist(df['Hizmet_Hizi'], bins=20)
    plt.xlabel('Hizmet Hızı')
    plt.ylabel('Frekans')
    plt.title('Hizmet Hızı Dağılımı')

    plt.subplot(2, 2, 3)
    plt.hist(df['Yemek_Kalitesi'], bins=20)
    plt.xlabel('Yemek Kalitesi')
    plt.ylabel('Frekans')
    plt.title('Yemek Kalitesi Dağılımı')

    plt.subplot(2, 2, 4)
    plt.hist(df['Musteri_Memnuniyeti'], bins=20)
    plt.xlabel('Müşteri Memnuniyeti')
    plt.ylabel('Frekans')
    plt.title('Müşteri Memnuniyeti Dağılımı')

    plt.tight_layout()
    plt.show()

    # 2. Korelasyon Analizi
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.show()

    # 3. Box Plotlar
    plt.figure(figsize=(15, 5))
    df.boxplot()
    plt.title('Değişkenlerin Dağılımı')
    plt.show()

    # 4. Kategorik Analiz
    df['Hizmet_Kategori'] = pd.cut(df['Hizmet_Hizi'],
                                   bins=[0, 3.33, 6.66, 10],
                                   labels=['Yavaş', 'Normal', 'Hızlı'])

    df['Kalite_Kategori'] = pd.cut(df['Yemek_Kalitesi'],
                                   bins=[0, 3.33, 6.66, 10],
                                   labels=['Kötü', 'İyi', 'Mükemmel'])

    df['Memnuniyet_Kategori'] = pd.cut(df['Musteri_Memnuniyeti'],
                                       bins=[0, 3.33, 6.66, 10],
                                       labels=['Düşük', 'Orta', 'Yüksek'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    df['Hizmet_Kategori'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0])
    axes[0].set_title('Hizmet Hızı Kategorileri')

    df['Kalite_Kategori'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title('Yemek Kalitesi Kategorileri')

    df['Memnuniyet_Kategori'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[2])
    axes[2].set_title('Müşteri Memnuniyeti Kategorileri')

    plt.tight_layout()
    plt.show()


def create_sensitivity_analysis():
    """Hassasiyet analizi yapar ve görselleştirir"""
    service_range = np.arange(0, 10.1, 0.5)
    quality_range = np.arange(0, 10.1, 0.5)

    results = []
    for service in service_range:
        for quality in quality_range:
            satisfaction_simulator.input['service_speed'] = service
            satisfaction_simulator.input['food_quality'] = quality
            satisfaction_simulator.compute()
            results.append([service, quality, satisfaction_simulator.output['customer_satisfaction']])

    sensitivity_df = pd.DataFrame(results, columns=['Hizmet_Hizi', 'Yemek_Kalitesi', 'Musteri_Memnuniyeti'])

    plt.figure(figsize=(10, 8))
    pivot_table = sensitivity_df.pivot(index='Hizmet_Hizi',
                                       columns='Yemek_Kalitesi',
                                       values='Musteri_Memnuniyeti')

    sns.heatmap(pivot_table, cmap='viridis', center=5)
    plt.title('Hassasiyet Analizi')
    plt.xlabel('Yemek Kalitesi')
    plt.ylabel('Hizmet Hızı')
    plt.show()


def analyze_data(df):
    """İstatistiksel analiz yapar ve sonuçları yazdırır"""
    print("\nVeri Seti İstatistikleri:")
    print(df.describe())

    print("\nKategori Bazlı Analiz:")
    for col in ['Hizmet_Kategori', 'Kalite_Kategori', 'Memnuniyet_Kategori']:
        print(f"\n{col} Dağılımı:")
        print(df[col].value_counts(normalize=True).multiply(100).round(2))


def save_results(df):
    """Sonuçları CSV dosyasına kaydeder"""
    df.to_csv('musteri_memnuniyeti_verileri.csv', index=False)
    print("\nVeriler 'musteri_memnuniyeti_verileri.csv' dosyasına kaydedildi.")


def load_and_preprocess_zomato_data():
    """Zomato veri setini yükler ve ön işler"""
    try:
        df = pd.read_csv('zomato.csv', encoding='latin1')
        print("\nCSV dosyasındaki sütunlar:")
        print(df.columns.tolist())

        # Gerçek sütun isimlerini kullan
        selected_columns = {
            'Aggregate rating': 'Musteri_Memnuniyeti',
            'Average Cost for two': 'Maliyet',
            'Votes': 'Oy_Sayisi',
            'Cuisines': 'Mutfak_Turu',
            'Has Table booking': 'Rezervasyon',
            'Has Online delivery': 'Online_Teslimat'
        }

        # Seçili sütunları al ve NaN değerleri temizle
        df = df[selected_columns.keys()].rename(columns=selected_columns)
        df = df.dropna()  # NaN değerleri temizle

        # Veri temizleme ve normalleştirme
        df['Musteri_Memnuniyeti'] = pd.to_numeric(df['Musteri_Memnuniyeti'], errors='coerce') * 2
        df['Maliyet'] = pd.to_numeric(df['Maliyet'], errors='coerce')
        df['Oy_Sayisi'] = pd.to_numeric(df['Oy_Sayisi'], errors='coerce')

        # NaN değerleri temizle
        df = df.dropna()

        # Maliyet kategorileri oluştur
        df['Maliyet_Kategori'] = pd.qcut(df['Maliyet'],
                                         q=3,
                                         labels=['Ekonomik', 'Orta', 'Pahalı'])

        # Temel istatistikler
        print("\nZomato Veri Seti İstatistikleri:")
        print(f"Toplam Restoran Sayısı: {len(df)}")
        print(f"Ortalama Müşteri Memnuniyeti: {df['Musteri_Memnuniyeti'].mean():.2f}/10")
        print(f"Ortalama Oy Sayısı: {df['Oy_Sayisi'].mean():.0f}")

        return df
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None

def analyze_real_data(df):
    """Gerçek veri seti için detaylı analiz"""
    try:
        plt.figure(figsize=(15, 10))

        # 1. Müşteri Memnuniyeti Dağılımı
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='Musteri_Memnuniyeti', bins=20)
        plt.title('Müşteri Memnuniyeti Dağılımı')

        # 2. Fiyat vs Memnuniyet İlişkisi
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='Maliyet_Kategori', y='Musteri_Memnuniyeti')
        plt.title('Fiyat Kategorilerine Göre Memnuniyet')
        plt.xticks(rotation=45)

        # 3. Oy Sayısı vs Memnuniyet
        plt.subplot(2, 2, 3)
        plt.scatter(df['Oy_Sayisi'], df['Musteri_Memnuniyeti'], alpha=0.5)
        plt.xlabel('Oy Sayısı')
        plt.ylabel('Müşteri Memnuniyeti')
        plt.title('Oy Sayısı ve Memnuniyet İlişkisi')

        # 4. Online Teslimat Etkisi
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='Online_Teslimat', y='Musteri_Memnuniyeti')
        plt.title('Online Teslimat ve Memnuniyet İlişkisi')

        plt.tight_layout()
        plt.show()

        # Ek analizler
        create_additional_insights(df)
    except Exception as e:
        print(f"Görselleştirme sırasında hata oluştu: {str(e)}")





def create_additional_insights(df):
    """Ek içgörüler oluşturur"""
    # Korelasyon Matrisi
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Musteri_Memnuniyeti', 'Maliyet', 'Oy_Sayisi']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Değişkenler Arası Korelasyon')
    plt.show()

    # Maliyet Kategorilerine Göre Dağılım
    plt.figure(figsize=(10, 5))
    df['Maliyet_Kategori'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Restoranların Maliyet Kategorilerine Göre Dağılımı')
    plt.show()


def compare_datasets(synthetic_df, real_df):
    """Sentetik ve gerçek veri setlerini karşılaştırır"""
    plt.figure(figsize=(15, 10))

    # 1. Müşteri Memnuniyeti Karşılaştırması
    plt.subplot(2, 2, 1)
    plt.hist(synthetic_df['Musteri_Memnuniyeti'], bins=20, alpha=0.5, label='Sentetik Veri')
    plt.hist(real_df['Musteri_Memnuniyeti'], bins=20, alpha=0.5, label='Gerçek Veri')
    plt.xlabel('Müşteri Memnuniyeti')
    plt.ylabel('Frekans')
    plt.title('Müşteri Memnuniyeti Dağılımı Karşılaştırması')
    plt.legend()

    # 2. İstatistiksel Özet
    plt.subplot(2, 2, 2)
    stats_data = {
        'Metrik': ['Ortalama', 'Medyan', 'Std', 'Min', 'Max'],
        'Sentetik': [
            synthetic_df['Musteri_Memnuniyeti'].mean(),
            synthetic_df['Musteri_Memnuniyeti'].median(),
            synthetic_df['Musteri_Memnuniyeti'].std(),
            synthetic_df['Musteri_Memnuniyeti'].min(),
            synthetic_df['Musteri_Memnuniyeti'].max()
        ],
        'Gerçek': [
            real_df['Musteri_Memnuniyeti'].mean(),
            real_df['Musteri_Memnuniyeti'].median(),
            real_df['Musteri_Memnuniyeti'].std(),
            real_df['Musteri_Memnuniyeti'].min(),
            real_df['Musteri_Memnuniyeti'].max()
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    plt.axis('off')
    table = plt.table(cellText=stats_df.values,
                      colLabels=stats_df.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('İstatistiksel Karşılaştırma')

    # 3. Box Plot Karşılaştırması
    plt.subplot(2, 2, 3)
    box_data = [synthetic_df['Musteri_Memnuniyeti'], real_df['Musteri_Memnuniyeti']]
    plt.boxplot(box_data, labels=['Sentetik Veri', 'Gerçek Veri'])
    plt.ylabel('Müşteri Memnuniyeti')
    plt.title('Müşteri Memnuniyeti Box Plot Karşılaştırması')

    # 4. Yoğunluk Grafiği
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=synthetic_df, x='Musteri_Memnuniyeti', label='Sentetik Veri')
    sns.kdeplot(data=real_df, x='Musteri_Memnuniyeti', label='Gerçek Veri')
    plt.xlabel('Müşteri Memnuniyeti')
    plt.ylabel('Yoğunluk')
    plt.title('Müşteri Memnuniyeti Yoğunluk Karşılaştırması')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # İstatistiksel karşılaştırma sonuçlarını yazdır
    print("\nKarşılaştırmalı İstatistikler:")
    print("\nSentetik Veri:")
    print(synthetic_df['Musteri_Memnuniyeti'].describe())
    print("\nGerçek Veri:")
    print(real_df['Musteri_Memnuniyeti'].describe())

def run_complete_analysis():
    """Tüm analiz sürecini çalıştırır"""
    print("Analiz başlıyor...")

    # 1. Sentetik Veri Analizi
    print("\n1. Sentetik Veri Analizi:")
    synthetic_df = generate_and_analyze_data(500)
    create_visualizations(synthetic_df)
    create_sensitivity_analysis()

    # 2. Gerçek Veri Analizi
    print("\n2. Gerçek Veri Analizi:")
    real_df = load_and_preprocess_zomato_data()
    if real_df is not None:
        analyze_real_data(real_df)

        # 3. Karşılaştırmalı Analiz
        print("\n3. Karşılaştırmalı Analiz:")
        compare_datasets(synthetic_df, real_df)

    # Sonuçları kaydet
    synthetic_df.to_csv('sentetik_veri_sonuclari.csv', index=False)
    if real_df is not None:
        real_df.to_csv('gercek_veri_sonuclari.csv', index=False)

    print("\nAnaliz tamamlandı!")


if __name__ == "__main__":
    try:
        run_complete_analysis()
    except Exception as e:
        print(f"Program çalıştırılırken hata oluştu: {str(e)}")