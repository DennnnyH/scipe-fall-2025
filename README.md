# Fall 2025 SCIPE Research

Clustering & Anomaly Detection with Machine Learning

Project Summary

This project applies machine learning to network traffic data with a strong focus on privacy, model robustness, and critical evaluation of results. It combines IP address obfuscation, unsupervised clustering, and supervised anomaly detection to explore user behavior patterns while minimizing sensitive data exposure.

The work emphasizes not just model performance, but why certain results should or should not be trusted.

Key Features

🔒 Privacy-first preprocessing via IP address obfuscation

📊 Unsupervised network clustering using PCA, t-SNE, and K-Means

🚨 Anomaly detection with XGBoost and neural networks

📈 Comprehensive evaluation using accuracy, F1, ROC curves, confusion matrices, and silhouette scores

🧠 Critical analysis of overfitting and data limitations

IP Address Obfuscation

To protect sensitive information and reduce the risk of overfitting:

The 4th octet of IP addresses is zeroed out

Example: 10.0.3.27 → 10.0.3.0

This preserves subnet-level structure while preventing identification of individual hosts

Helps prevent models from memorizing exact IP addresses

Network Clustering
Approach

Dimensionality Reduction

Principal Component Analysis (PCA)

t-SNE for visualization

Clustering

K-Means

Findings

t-SNE visualizations revealed 6 distinct clusters

Cluster quality evaluated using silhouette scores

Cluster centers were analyzed to understand dominant feature contributions

Limitations

The dataset did not include labels for user roles, device types, or applications

While structural clusters were found, they could not be confidently mapped to real-world entities

This section demonstrates practical experience with unsupervised learning and awareness of its interpretability limits.

Anomaly Detection
XGBoost Classifier

Achieved near-perfect accuracy and F1 scores

Results were considered suspiciously high

Potential causes:

Overfitting

Data leakage

Strong class separability in the dataset

These results were intentionally not taken at face value.

Multilayer Perceptron (MLP)

Baseline models showed low initial performance

Performance improved through:

Hyperparameter tuning

Data resampling to address class imbalance

Results

Best tuned model improved accuracy and F1 scores by ~40%

Evaluated using:

Confusion matrices

ROC curves

This model produced more realistic and trustworthy performance characteristics.

Technologies Used

Python

scikit-learn

XGBoost

PyTorch / TensorFlow (use whichever you actually used)

NumPy / Pandas / Matplotlib / Seaborn

What This Project Demonstrates

Ability to design privacy-aware ML pipelines

Hands-on experience with unsupervised and supervised learning

Strong understanding of model evaluation and overfitting risks

Willingness to question “perfect” results rather than blindly report them

Clear communication of technical findings and limitations

Future Improvements

Incorporate datasets with device or user-role labels

Explore unsupervised anomaly detection (autoencoders, isolation forests)

Perform cross-dataset generalization testing

Add automated data leakage checks
