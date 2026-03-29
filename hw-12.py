import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pygad

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

# 3.1.1. Дані
data = load_breast_cancer()
X = data.data
y = data.target  # 0 - malignant, 1 - benign

# 3.1.2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.2. Кластеризація трьома методами

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


# 4. Код для PCA та аналізу дисперсії

# 4.2. Зменшення розмірності даних за допомогою метода PCA до 2 компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4.3. Частка поясненої дисперсії
explained = pca.explained_variance_ratio_

print("Explained variance ratio:", explained)
print("Total explained variance:", explained.sum())

# 4.3. Візуалізація PCA з кольорами за класами
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


# 5. Розподіл класів у просторі PC1–PC2

# Побудова діаграми
plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap='coolwarm',
    alpha=0.7,
    s=40
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Розподіл класів у просторі перших двох головних компонент PCA')
plt.colorbar(label='target (0 = malignant, 1 = benign)')
plt.show()


# 6. Логістична регресія на Breast Cancer

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Модель логістичної регресії
log_reg = LogisticRegression(max_iter=500, solver='lbfgs')
log_reg.fit(X_train, y_train)

# Прогнози
y_pred = log_reg.predict(X_test)

# Оцінка
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)


# 7.1. Підготовка: train/test
# X_scaled, y вже є з попередніх кроків
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7.2. Базові функції для логістичної регресії
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict_label(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)

def logistic_loss(X, y, w, b):
    p = predict_proba(X, w, b)
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# 7.3. Різні методи спуску

# 7.3.1. Full-batch gradient descent
def train_gd_full(X, y, lr=0.1, n_epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(n_epochs):
        p = predict_proba(X, w, b)
        error = p - y

        grad_w = X.T @ error / n_samples
        grad_b = np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

w_full, b_full = train_gd_full(X_train, y_train, lr=0.1, n_epochs=500)
y_pred_full = predict_label(X_test, w_full, b_full)
acc_full = accuracy_score(y_test, y_pred_full)
print("Full-batch GD accuracy:", acc_full)

# 7.3.2. Stochastic gradient descent (SGD)
def train_gd_sgd(X, y, lr=0.01, n_epochs=20):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    rng = np.random.default_rng(42)

    for epoch in range(n_epochs):
        indices = rng.permutation(n_samples)
        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]

            p = predict_proba(xi, w, b)
            error = p - yi

            grad_w = xi.T @ error
            grad_b = np.mean(error)

            w -= lr * grad_w
            b -= lr * grad_b

    return w, b

w_sgd, b_sgd = train_gd_sgd(X_train, y_train, lr=0.01, n_epochs=20)
y_pred_sgd = predict_label(X_test, w_sgd, b_sgd)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("SGD accuracy:", acc_sgd)

# 7.3.3. Mini-batch gradient descent
def train_gd_minibatch(X, y, lr=0.05, n_epochs=100, batch_size=32):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    rng = np.random.default_rng(42)

    for epoch in range(n_epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]

            p = predict_proba(xb, w, b)
            error = p - yb

            grad_w = xb.T @ error / len(xb)
            grad_b = np.mean(error)

            w -= lr * grad_w
            b -= lr * grad_b

    return w, b

w_mb, b_mb = train_gd_minibatch(X_train, y_train, lr=0.05, n_epochs=100, batch_size=32)
y_pred_mb = predict_label(X_test, w_mb, b_mb)
acc_mb = accuracy_score(y_test, y_pred_mb)
print("Mini-batch GD accuracy:", acc_mb)


# 8. Логістична регресія + Генетичний алгоритм (GA)

# 8.1. Підготовка даних
# X_scaled, y вже існують з попередніх кроків
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8.2. Логістична регресія (ручні функції)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba_ga(X, w):
    return sigmoid(X @ w)

def predict_label_ga(X, w):
    return (predict_proba_ga(X, w) >= 0.5).astype(int)

# 8.3. Функція пристосованості (fitness)
#GA максимізує fitness, тому ми беремо accuracy як ціль.
def fitness_func(ga_instance, solution, solution_idx):
    w = solution
    y_pred = predict_label_ga(X_train, w)
    acc = accuracy_score(y_train, y_pred)
    return acc

# 8.4. Налаштування та запуск GA
"""
Використовуємо ті самі параметри, які були у минулому ДЗ про GA‑класифікатор:
  - популяція 50
  - 100 поколінь
  - одноточковий кросовер
  - мутація з невеликим відхиленням
  - елітизм
"""

num_features = X_train.shape[1]

ga = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=num_features,
    init_range_low=-1.0,
    init_range_high=1.0,
    mutation_percent_genes=10,
    mutation_type="random",
    mutation_by_replacement=False,
    crossover_type="single_point",
    parent_selection_type="tournament",
    keep_parents=2
)

ga.run()

# 8.5. Отримання найкращого розв’язку
solution, solution_fitness, solution_idx = ga.best_solution()
print("Best fitness (train accuracy):", solution_fitness)

# 8.6. Оцінка на тестовій вибірці
y_pred_test = predict_label_ga(X_test, solution)
test_acc = accuracy_score(y_test, y_pred_test)

print("Test accuracy (GA):", test_acc)
