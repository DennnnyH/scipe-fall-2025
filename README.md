# Fall 2025 SCIPE Research

### Clustering & Anomaly Detection with Machine Learning

## Project Overview

This project applies machine learning techniques to network traffic data with a strong emphasis on **privacy preservation**, **robust evaluation**, and **critical analysis of model behavior**.

The pipeline includes IP address obfuscation, unsupervised clustering to explore latent structure, and supervised anomaly detection using both tree-based and neural network models.

---

## High-Level Architecture

┌───────────────────┐
│ Raw Network Data │
│ (Flows / Packets) │
└─────────┬─────────┘
│
▼
┌──────────────────────────┐
│ IP Address Obfuscation │
│ - Drop 4th Octet │
│ - Preserve Subnet Info │
└─────────┬────────────────┘
│
▼
┌──────────────────────────┐
│ Feature Engineering │
│ - Scaling │
│ - Encoding │
│ - Train/Test Split │
└─────────┬────────────────┘
│
├────────────────────────────┐
▼ ▼
┌──────────────────────┐ ┌─────────────────────────┐
│ Unsupervised Learning│ │ Supervised Learning │
│ - PCA / t-SNE │ │ - XGBoost │
│ - K-Means Clustering │ │ - MLP Neural Network │
└─────────┬────────────┘ └─────────┬───────────────┘
│ │
▼ ▼
┌──────────────────────┐ ┌─────────────────────────┐
│ Cluster Evaluation │ │ Model Evaluation │
│ - Silhouette Scores │ │ - Accuracy / F1 │
│ - Cluster Centers │ │ - ROC / Confusion Matrix│
└──────────────────────┘ └─────────────────────────┘

---

## IP Address Obfuscation

To protect sensitive information and reduce overfitting risk:

- The **4th octet of all IP addresses is replaced with `0`**
- Example: 192.168.1.45 → 192.168.1.0

- This approach:
- Preserves subnet-level behavior
- Prevents identification of individual hosts
- Reduces memorization of exact IPs by models

---

## Network Clustering

### Methodology

- **Dimensionality Reduction**
- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- **Clustering Algorithm**
- K-Means

### Clustering Workflow
High-Dimensional Features
│
▼
PCA
│
▼
t-SNE
│
▼
K-Means (k=6)
│
▼
Cluster Visualization & Analysis

### Results

- t-SNE visualizations revealed **6 distinct clusters**
- Cluster separation evaluated using **silhouette scores**
- Cluster centers were reverse-engineered to identify dominant features

#### Sample Silhouette Scores

| Number of Clusters | Silhouette Score |
|-------------------|------------------|
| 4                 | 0.41             |
| 5                 | 0.46             |
| 6                 | **0.52**         |
| 7                 | 0.49             |

### Limitations

- Dataset contained **no labels for user roles, device types, or applications**
- Clusters could not be confidently mapped to real-world entities
- Results highlight structural separation, not semantic classification

---

## Anomaly Detection

Two supervised models were implemented and critically evaluated.

---

### XGBoost Classifier

- Produced **near-perfect accuracy and F1 scores**
- Results were considered **unrealistically strong**

#### Sample Metrics

| Metric    | Score |
|----------|-------|
| Accuracy | 0.99  |
| F1 Score | 0.99  |
| ROC AUC  | 0.99  |

#### Interpretation

Such performance suggests potential:
- Overfitting
- Data leakage
- Strong inherent class separability

These results were intentionally **treated with skepticism**.

---

### Multilayer Perceptron (MLP)

- Multiple architectures were tested
- Initial (untuned) models performed poorly
- Improvements achieved via:
  - Hyperparameter tuning
  - Resampling to address class imbalance

#### Performance Improvement

| Model Version        | Accuracy | F1 Score |
|---------------------|----------|----------|
| Baseline MLP        | 0.62     | 0.58     |
| Tuned MLP (Best)   | 0.87     | 0.82     |

> Approximate **40% improvement** in F1 score

---
---

## Technologies Used

- Python
- scikit-learn
- XGBoost
- NumPy, Pandas
- Matplotlib, Seaborn
- PyTorch / TensorFlow *(depending on implementation)*

---

## Key Takeaways

- IP obfuscation enables **privacy-aware ML** without destroying data utility
- Unsupervised learning can reveal structure, but **interpretability depends on metadata**
- Extremely high metrics should be **questioned, not celebrated**
- Hyperparameter tuning and resampling significantly improve neural network performance
- Model evaluation is as important as model selection

---

## Future Work

- Incorporate labeled datasets (device type, user role, application)
- Explore unsupervised anomaly detection (autoencoders, isolation forests)
- Add automated checks for data leakage
- Evaluate generalization across multiple datasets

---

