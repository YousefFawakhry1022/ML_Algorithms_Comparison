# ML Algorithms Comparison: Default Prediction

A compact machine learning pipeline comparing Support Vector Machines (SVM), Decision Trees, and Random Forests for classification (default prediction). 

## 🚀 Quick Overview
- **Objective:** Predict default status using structured data.
- **Algorithms:** Support Vector Machine (SVC), Decision Tree Classifier, Random Forest Classifier.
- **Techniques Used:** Data Downsampling, One-Hot Encoding, Hyperparameter Tuning (`GridSearchCV`), Cost Complexity Pruning.

## 🧠 Methodology
1. **Data Preprocessing:** Handled missing data, applied One-Hot Encoding (`pd.get_dummies`), and downsampled the dataset (1,000 samples per class) to optimize SVM training performance.
2. **SVM Optimization:** Fine-tuned `C` and `gamma` parameters using `GridSearchCV` with an RBF kernel.
3. **Decision Tree Pruning:** Applied Cost Complexity Pruning (extracting optimal `ccp_alpha`) to prevent the tree from overfitting.
4. **Random Forest:** Trained a robust baseline ensemble classifier for direct comparison.

## 🛠️ Tech Stack
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Environment:** Jupyter Notebook / Miniconda
