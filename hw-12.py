import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix


# 1. Завантаження Breast Cancer Wisconsin (Diagnostic)
# Завантаження набору даних
data = load_breast_cancer()

# Ознаки та ціль
X = data.data
y = data.target

# Перетворимо у DataFrame для зручності аналізу
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

df.head()

df.info()
df.describe().T

# 2. Код для побудови pairplot

# Обираємо найбільш інформативні ознаки
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]

# Створюємо pairplot
sns.pairplot(
    df[selected_features + ['target']],
    hue='target',
    diag_kind='kde',
    palette='coolwarm',
    plot_kws={'alpha': 0.6, 's': 40}
)

plt.show()


# 3. Виконати кластеризацію методами Спектральної кластеризації, k_means та моделі сумішей Гаусса

# 3.1. Підготовка даних та масштабування
from sklearn.preprocessing import StandardScaler

# 3.1.1. Дані
data = load_breast_cancer()
X = data.data
y = data.target  # 0 - malignant, 1 - benign

# 3.1.2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.2. Кластеризація трьома методами
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

# 3.2.1. KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3.2.2. Spectral Clustering
spectral = SpectralClustering(
    n_clusters=2,
    affinity='rbf',
    assign_labels='kmeans',
    random_state=42
)
spectral_labels = spectral.fit_predict(X_scaled)

# 3.2.3. Gaussian Mixture
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# 3.3. Порівняння кластерів з істинними класами
"""
Оскільки кластери не мають «імен», ми не дивимось на точність напряму, а використовуємо метрики, інваріантні до перестановки міток:

    ARI (Adjusted Rand Index)

    NMI (Normalized Mutual Information)

    Плюс — таблиця відповідності (contingency table).
"""
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

def evaluate_clustering(labels, y_true, name):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    cm = confusion_matrix(y_true, labels)
    print(f'=== {name} ===')
    print(f'ARI: {ari:.4f}')
    print(f'NMI: {nmi:.4f}')
    print('Contingency table (rows: true classes, cols: clusters):')
    print(cm)
    print()

evaluate_clustering(kmeans_labels, y, 'KMeans')
evaluate_clustering(spectral_labels, y, 'Spectral Clustering')
evaluate_clustering(gmm_labels, y, 'Gaussian Mixture')


# 4. Зменшення розмірності даних за допомогою метода PCA

# 4.1. Код для PCA та аналізу дисперсії
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# PCA до 2 компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Частка поясненої дисперсії
explained = pca.explained_variance_ratio_

print("Explained variance ratio:", explained)
print("Total explained variance:", explained.sum())

# 4.2. Візуалізація PCA з кольорами за класами
plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y, cmap='coolwarm', alpha=0.7, s=40
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: перші дві головні компоненти')
plt.colorbar(label='target (0=malignant, 1=benign)')
plt.show()
