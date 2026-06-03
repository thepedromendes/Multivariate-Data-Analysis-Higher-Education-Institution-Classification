# Multivariate Data Analysis: Higher Education Institution Classification

This repository contains the computational project developed for the **Data Analysis Statistics (EAD)** course. The primary objective is to explore, segment, and classify higher education institutions as **Public** or **Private** (`bl_private`) based on academic, demographic, and financial indicators.

The workflow covers everything from data preprocessing and exploratory analysis to the application of advanced multivariate statistical techniques, encompassing both unsupervised methods and supervised predictive modeling.

---

## Implemented Methods & Techniques

The project is built upon four major statistical pillars:

1. **Exploratory and Bivariate Analysis:**
   * Calculation of comprehensive descriptive statistics (location, dispersion, and shape indicators via `moments`).
   * Bivariate hypothesis testing for group comparison (Welch's $t$-test and Wilcoxon Rank-Sum test).
   * Parametric correlation analysis featuring a visual correlation matrix plot (`corrplot`).

2. **Dimensionality Reduction (Unsupervised):**
   * **PCA (Principal Component Analysis):** Reducing the feature space and visualizing total explained variance using *Scree Plots* and factor *loadings* analysis.

3. **Clustering:**
   * **K-Means Clustering:** Determining the optimal number of clusters using the *Elbow* (WSS) and *Silhouette* methods, and benchmarking the resulting clusters against the true nature of the institutions.

4. **Supervised Classification Models:**
   * **LDA (Linear Discriminant Analysis):** A linear decision boundary assuming homoscedasticity.
   * **QDA (Quadratic Discriminant Analysis):** A flexible quadratic boundary tailored for distinct covariance matrices.
   * **Logistic Regression (LR):** Probabilistic modeling with the calculation and interpretation of *Odds Ratios* (95% CI).
   * **KNN ($K$-Nearest Neighbors):** A non-parametric algorithm with the hyperparameter $K$ optimized via cross-validation.

---

## Repository Structure

To ensure the reproducibility of the script, please make sure you maintain the following folder structure in your working directory:

```text
├── EAD_project.R              # Main R script
├── EAD_report.pdf
├── Data/
│   └── volis_dataset.csv      # Original dataset containing the institutions
├── figures/                   # Automatically generated: Charts and ROC Curves
└── outputs/                   # Automatically generated: CSV tables and confusion matrices
