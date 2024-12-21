import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzy_logic import satisfaction_simulator


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

    # Veriyi kaydet
    df.to_csv('musteri_memnuniyeti_verileri.csv', index=False)
    return df


def visualize_data(df):
    """Veri görselleştirme"""
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


def create_correlation_analysis(df):
    """Korelasyon analizi"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.show()
