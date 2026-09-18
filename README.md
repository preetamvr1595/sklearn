# Scikit-Learn Mastery Series 🚀

A comprehensive, production-grade 5-part repository for mastering machine learning algorithms, preprocessing techniques, hyperparameter tuning, model evaluation, and deployment with **Scikit-Learn**.

---

## 📚 Curriculum & Notebook Overview

### 1. [`01_sklearn_basics.ipynb`](01_sklearn_basics.ipynb)
- **Core Concepts**: Scikit-Learn Estimator API design (`fit`, `transform`, `predict`).
- **Data Handling**: Loading standard datasets (`load_iris`, `load_wine`), Pandas DataFrame conversions, and Exploratory Data Analysis (EDA).
- **Preprocessing**: Feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), categorical encoding (`OneHotEncoder`, `OrdinalEncoder`), and missing value imputation (`SimpleImputer`).
- **Pipelines**: Building leak-free data processing pipelines using `Pipeline` and `ColumnTransformer`.

---

### 2. [`02_classification.ipynb`](02_classification.ipynb)
- **Supervised Classification Algorithms**: Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting.
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC curves, Precision-Recall (PR) curves, and Log Loss.
- **Imbalanced Datasets**: Class weight adjustments, SMOTE oversampling, and threshold tuning.
- **Ensemble Techniques**: Voting Classifiers, Stacking, and Bagging.

---

### 3. [`03_regression.ipynb`](03_regression.ipynb)
- **Supervised Regression Algorithms**: Ordinary Least Squares (OLS) Linear Regression, Ridge ($L_2$), Lasso ($L_1$), ElasticNet, Decision Tree Regressor, and Random Forest Regressor.
- **Evaluation Metrics**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), $R^2$ Score, and Adjusted $R^2$.
- **Diagnostics**: Residual plots, homoscedasticity checks, and polynomial feature expansion.

---

### 4. [`04_unsupervised_preprocessing.ipynb`](04_unsupervised_preprocessing.ipynb)
- **Clustering**: K-Means (`k-means++`), Agglomerative Hierarchical Clustering (Ward/Complete/Average linkages with Dendrograms), and DBSCAN density clustering.
- **Clustering Evaluation**: Elbow Method (Inertia), Silhouette Analysis (per-sample plots), Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Calinski-Harabasz, and Davies-Bouldin indices.
- **Dimensionality Reduction**: Principal Component Analysis (PCA), TruncatedSVD for sparse data, and t-SNE manifold visualization across perplexities.
- **Advanced Feature Selection**: Filter (`SelectKBest`, ANOVA F-test, Mutual Info), Wrapper (`RFE`), and Embedded L1 (`SelectFromModel`).
- **Advanced Imputation & Outliers**: `KNNImputer`, `IterativeImputer` (MICE), `IsolationForest`, and `LocalOutlierFactor` (LOF).

---

### 5. [`05_model_selection_project.ipynb`](05_model_selection_project.ipynb)
- **Hyperparameter Optimization**: Exhaustive `GridSearchCV` vs randomized continuous search `RandomizedSearchCV`.
- **Diagnostic Curves**: Learning curves for diagnosing bias/variance (underfitting vs overfitting) and validation curves.
- **Heterogeneous Pipelines**: `ColumnTransformer` processing mixed numeric and categorical variables simultaneously without data leakage.
- **Capstone Real-World Project**: End-to-end model training, test set validation, confusion matrix visualization, and model serialization using `joblib`.

---

## 🛠️ Requirements & Setup

```bash
# Clone repository
git clone https://github.com/preetamvr1595/sklearn.git
cd sklearn

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
```

---

## 🤝 Author
Created by **Preetham** as part of the **Scikit-Learn Mastery Series**.