# Fall 2025 SCIPE Research 

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/DennnnyH/anomaly-detection)

## Project Overview

This repository contains a series of experiments applying machine learning techniques to network traffic data for clustering and anomaly detection. The project emphasizes a robust and critical approach, prioritizing privacy preservation, thorough model evaluation, and a skeptical analysis of model performance.

The end-to-end pipeline involves:
1.  **Privacy-Preserving Preprocessing**: Obfuscating IP addresses to protect user privacy while retaining valuable subnet-level information.
2.  **Unsupervised Clustering**: Using dimensionality reduction (PCA, t-SNE) and K-Means to explore the latent structure of the network data without labels.
3.  **Supervised Anomaly Detection**: Training and critically evaluating both tree-based (XGBoost) and neural network (MLP) models to classify traffic flows.

## High-Level Architecture

The project follows a structured data processing and modeling pipeline:

```
┌───────────────────┐
│ Raw Network Data  │
│ (Netflow Records) │
└─────────┬─────────┘
          │
          ▼
┌──────────────────────────┐
│ IP Address Obfuscation   │
│ - Drop 4th Octet         │
│ - Convert to Numeric     │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Feature Engineering      │
│ - Scaling & Encoding     │
│ - Train/Test Split       │
│ - Class Imbalance Handling│
└─────────┬────────────────┘
          │
├─────────────────┬─────────────────┐
▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│ Unsupervised    │ │ Supervised      │
│ Learning        │ │ Learning        │
│ - PCA / t-SNE   │ │ - XGBoost       │
│ - K-Means       │ │ - MLP Network   │
└─────────┬───────┘ └───────┬─────────┘
          │                 │
          ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│ Cluster         │ │ Model           │
│ Evaluation      │ │ Evaluation      │
│ - Silhouette    │ │ - Accuracy / F1 │
│ - Visualization │ │ - ROC / P-R Curve│
└─────────────────┘ └─────────────────┘
```

## Privacy-Preserving IP Obfuscation

To prevent the identification of individual hosts and reduce the risk of models memorizing specific IP addresses, a simple yet effective obfuscation technique was applied:

-   The **4th octet of all source and destination IP addresses is replaced with `0`**.
    -   *Example*: `192.168.1.123` → `192.168.1.0`
-   This method preserves subnet-level information, which is often crucial for identifying network-wide behavior patterns.
-   The resulting obfuscated IP addresses (e.g., `192.168.1.0`) are then converted into a single numeric (integer/float) representation for use as a model feature.

## Unsupervised Learning: Network Clustering

Before attempting supervised detection, unsupervised methods were used to identify natural groupings within the data.

### Methodology
1.  **Feature Selection**: A subset of numerical features like `FLOW_DURATION`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, and `OUT_PKTS` were selected.
2.  **Data Scaling**: `StandardScaler` was used to normalize features to have zero mean and unit variance, preventing features with larger scales from dominating the clustering process.
3.  **Dimensionality Reduction**:
    -   **PCA (Principal Component Analysis)** was used to reduce the feature space and enable 2D visualization.
    -   **t-SNE (t-distributed Stochastic Neighbor Embedding)** was applied to a sample of the data to create more refined 2D visualizations that emphasize local structure.
4.  **Clustering**: The **K-Means** algorithm was applied to the dimensionally-reduced data to partition it into distinct clusters.

### Results and Analysis
-   Visualizations using both PCA and t-SNE revealed the presence of several distinct clusters in the data.
-   Silhouette scores were calculated to quantitatively assess cluster separation for different values of *k* (number of clusters). The analysis suggested that **k=6** provided a good balance of cluster cohesion and separation.
-   By applying an `inverse_transform` on the cluster centers, we could analyze the original feature values that characterized each group, providing insights into what types of traffic defined each cluster (e.g., one cluster for short, low-byte flows, another for long-duration, high-byte flows).

## Supervised Learning: Anomaly Detection

Two different supervised models were trained to classify network flows into three categories: Normal (0), Reconnaissance (1), and BruteForce (2).

### 1. XGBoost Classifier

An XGBoost model was trained on the dataset, yielding suspiciously high performance metrics.

| Metric | Score |
| :--- | :--- |
| Accuracy | 0.999 |
| F1 Score (Weighted) | 0.999 |
| ROC AUC (Avg) | 0.999 |

**Interpretation:** While seemingly ideal, these near-perfect scores were treated with **extreme skepticism**. Such results often point to issues like:
-   **Data Leakage**: A feature that inadvertently contains information about the target label.
-   **Overfitting**: The model has memorized the training data instead of learning general patterns.
-   **Trivial Separability**: The features chosen make the classes trivially easy to separate, which may not hold for real-world data.

This outcome serves as a critical lesson: "too good to be true" metrics should always trigger further investigation, not celebration.

### 2. Multilayer Perceptron (MLP)

A neural network approach was taken to provide a comparative model. The development process was iterative and highlights key machine learning practices.

1.  **Baseline Model**: An initial, untuned `MLPClassifier` performed poorly, achieving an accuracy of only around **0.72**. The model struggled significantly with the minority classes due to severe class imbalance.

2.  **Addressing Class Imbalance**: The dataset is heavily skewed towards the 'Normal' class. To address this, `RandomUnderSampler` from the `imbalanced-learn` library was used to create a balanced training set by down-sampling the majority class.

3.  **Tuned Model**: A deeper MLP with a `(128, 64, 32)` architecture was trained on the resampled data. This, combined with hyperparameter tuning (e.g., `learning_rate_init=0.005`), led to a significant and more realistic performance improvement.

#### Performance Improvement

| Model Version | Accuracy | F1 Score (Micro) |
| :--- | :--- | :--- |
| Baseline MLP (Imbalanced Data) | ~0.72 | ~0.72 |
| **Resampled & Tuned MLP** | **~0.87** | **~0.87** |

The final MLP model demonstrates strong, credible performance, with ROC AUC scores of **0.99, 0.88, and 0.99** for the three classes, respectively. This iterative process underscores the importance of handling class imbalance and proper tuning.

## Technologies Used

-   **Data Manipulation**: `pandas`, `numpy`
-   **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
-   **Visualization**: `matplotlib`

## Notebooks
-   `network_traffic_clustering_pca.ipynb`: K-Means clustering with PCA for visualization.
-   `network_traffic_clustering_tsne.ipynb`: K-Means clustering with t-SNE for visualization on a data sample.
-   `network_traffic_clustering_pca_more_params.ipynb`: Further clustering experiments with an expanded feature set.
-   `anomaly_detection_xgboost.ipynb`: Supervised classification using XGBoost.
-   `neural_network_anomoly_detection.ipynb`: Supervised classification using scikit-learn's MLPClassifier, including resampling techniques.
