# 🎓 Student Intervention System

> A data-driven machine learning pipeline to identify students at risk of academic failure — enabling early intervention before it's too late.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings from EDA](#key-findings-from-eda)
- [Model Results](#model-results)
- [Handling Class Imbalance with SMOTE](#handling-class-imbalance-with-smote)
- [Conclusions](#conclusions)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Overview

The **Student Intervention System** is a supervised machine learning project that predicts whether a secondary school student is at risk of failing the academic year. By leveraging a dataset of 395 students from two Portuguese public schools — Gabriel Pereira and Mousinho da Silveira — and analyzing 30 features spanning demographics, family background, lifestyle, and academic history, this project aims to support the development of early intervention systems.

Out of 395 students, 265 passed (67%) and 130 failed (33%), highlighting a meaningful class imbalance that the project explicitly addresses.

---

## Motivation

Academic failure has cascading long-term consequences — on employment, economic mobility, and mental health. Traditional systems often identify struggling students too late, after failure has already occurred. This project proposes a data-driven alternative: using observable behavioral and socioeconomic signals to flag at-risk students before the end of the academic year, giving schools the opportunity to intervene proactively.

The target class of interest is **failing students** (mapped as the positive class), with False Negatives considered the most critical error — an overlooked at-risk student who receives no support.

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Size:** 395 students, 30 features + 1 binary target (`passed` / `failed`)
- **Schools:** Gabriel Pereira (349 students) and Mousinho da Silveira (46 students), both public schools in Portugal
- **Reference article:** [Cortez & Silva, 2008 — Using Data Mining to Predict Secondary School Student Performance](https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf)

### Feature Summary

| Category | Features |
|---|---|
| Demographics | `sex`, `age`, `address` |
| Family Background | `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, `famrel`, `famsup` |
| Academic History | `failures`, `schoolsup`, `paid`, `studytime`, `absences` |
| Lifestyle | `romantic`, `goout`, `freetime`, `Dalc`, `Walc`, `activities` |
| Aspirations | `higher`, `internet`, `nursery`, `health`, `traveltime`, `reason` |

---

## Project Structure

```
Student_Intervention_System/
│
├── student-data.csv            # Raw dataset
├── correlation graph.png       # Pre-computed heatmap visualization
├── Student_Intervention_System.ipynb  # Full analysis notebook
└── README.md
```

---

## Methodology

The project follows a rigorous six-stage analytical pipeline:

### 1. Data Cleaning
- Verified zero null values across all 395 rows
- Applied **IQR-based outlier detection** on `absences` (upper fence = 20); edge cases were intentionally preserved to avoid discarding potentially meaningful data points
- Built a **correlation heatmap** using one-hot encoded features to identify key feature relationships

### 2. Exploratory Data Analysis
Three analytical lenses were applied:

**Parent background analysis** — family size, parental cohabitation status, mother/father education level (0–4), occupation, and guardian type were compared across passing and failing student groups.

**Factors contributing to passing** — study time, school support, family support, paid tutoring, internet access at home, and aspiration for higher education.

**Factors contributing to failing** — absences, prior class failures, extracurricular activities, romantic involvement, social activity (going out), and alcohol consumption.

### 3. Exploratory Data Analysis Questions
Targeted sub-group analyses included:
- Students with two teacher parents
- Students with low parental education (≤ 9th grade)
- Students with low study time + high free time + no higher education aspiration (low motivation proxy)
- Students in romantic relationships with high free time and low study time
- Effect of receiving paid tutoring vs. actual study investment
- Impact of combined school and family educational support
- Alcohol + social life interaction with passing rates

### 4. Data Preprocessing
- Dropped the `school` feature (treated as irrelevant to generalization)
- Applied `pd.get_dummies()` to convert all categorical variables into binary columns
- Mapped target variable: `no → 1` (at-risk), `yes → 0`
- 80/20 train-test split
- Applied **StandardScaler** for feature normalization

### 5. Model Training & Evaluation
Six classifiers were trained and evaluated using: Accuracy, Precision, Recall, F1-Score, ROC AUC, and Confusion Matrix.

### 6. Ensemble Learning & SMOTE
SMOTE was applied to address class imbalance, followed by a Voting Classifier ensemble combining all models with soft voting. Cross-validation was used for final model selection.

---

## Key Findings from EDA

**Parental Education has a measurable impact.** Students whose parents have a university-level education (level 4) showed an overall passing rate above 75%. This is consistent with literature on intergenerational educational transmission, particularly relevant given that the students were born around 1990, when higher education access outside major Portuguese cities was limited.

**Aspiration matters.** Students who expressed a desire for higher education showed significantly higher passing rates. Conversely, ~10% of failing students had no aspiration for further education, suggesting motivational disengagement as a risk factor.

**Prior failures are the strongest predictor.** The correlation heatmap and feature analysis consistently point to `failures` as the variable most strongly associated with the outcome — a student who has failed before is substantially more likely to fail again.

**Social lifestyle indicators compound risk.** Students who frequently went out (levels 4–5), consumed alcohol on both weekdays and weekends, and had high free time were concentrated in the failing group. The correlation between weekday and weekend alcohol consumption and outgoing behavior was noted in the heatmap.

**Gender patterns in study behavior.** Female students tended to show higher study time; male students showed higher alcohol consumption — consistent with national trends at the time.

**Guardian type correlates with age.** Students with an "other" guardian (i.e., not mother or father) tend to be older (18+), likely self-guardian, and show a notably lower passing rate.

---

## Model Results

All models were evaluated on a held-out test set (20%) with the class of interest being **failing students** (positive class = 1).

### Without SMOTE

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| Decision Tree | 60.76% | 51.43% | 56.25% | 53.73% | 60.04% |
| KNN | 68.35% | 50.00% | 36.00% | 41.86% | 59.67% |
| Logistic Regression | 67.09% | 62.50% | 46.88% | 53.57% | 63.86% |
| SVM | 59.49% | 50.00% | 3.12% | 5.88% | 50.50% |
| Neural Network (MLP) | 60.76% | 51.43% | 56.25% | 53.73% | 60.04% |
| Random Forest | 62.03% | 55.00% | 34.38% | 42.31% | 57.61% |

### With SMOTE

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| Decision Tree | 44.30% | 32.35% | 34.38% | 33.33% | 42.72% |
| KNN | 59.49% | 50.00% | 50.00% | 50.00% | 57.98% |
| Logistic Regression | 67.09% | 60.71% | 53.12% | 56.67% | 64.86% |
| **SVM** | **68.35%** | **61.29%** | **59.38%** | **60.32%** | **66.92%** |
| Neural Network (MLP) | 60.76% | 51.35% | 59.38% | 55.07% | 60.54% |
| Random Forest | 60.76% | 52.00% | 40.62% | 45.61% | 57.55% |

### Cross-Validation Scores (5-Fold)

| Model | CV Accuracy |
|---|---|
| Voting Classifier | 68.10% |
| Random Forest (no SMOTE) | 67.09% |
| Random Forest (SMOTE) | 68.86% |
| SVM (SMOTE) | 68.35% |

### Confusion Matrix Interpretation

In the context of student risk identification:
- **True Positive (TP):** Student predicted to fail and actually fails → Correctly flagged for intervention
- **False Negative (FN):** Student predicted to pass but actually fails → **Overlooked at-risk student** (most critical error)
- **False Positive (FP):** Student predicted to fail but actually passes → Unnecessary but harmless intervention
- **True Negative (TN):** Student predicted to pass and actually passes → Correctly identified

---

## Handling Class Imbalance with SMOTE

The original dataset has a 2:1 pass/fail ratio. Without correction, models were biased toward predicting the majority class, producing high False Negative rates — the most harmful outcome for an intervention system.

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples of the minority class (failing students) based on nearest-neighbor interpolation in feature space, balancing the training set to 218 samples per class. After applying SMOTE, **SVM with RBF kernel** showed the best balanced performance, particularly improving Recall (59.38%) — reducing the number of at-risk students who are missed.

---

## Conclusions

The analysis reveals that student academic performance is a multi-factor phenomenon. Social and behavioral indicators (alcohol consumption, going out, romantic involvement) compound underlying socioeconomic factors (parental education, family structure) to either protect or expose students to academic risk.

From a machine learning standpoint, the dataset's limited size (395 samples) and class imbalance make this a challenging classification task. No single model achieved dominant performance across all metrics. **SVM trained with SMOTE** offered the best overall profile for the system's goal of minimizing missed at-risk students, while **Random Forest with SMOTE** achieved strong cross-validated accuracy.

Future improvements could include: larger and more recent datasets, feature engineering on interaction variables, threshold tuning to maximize recall, and time-series data collection for longitudinal tracking.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Primary language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | ML models, preprocessing, evaluation |
| imbalanced-learn | SMOTE implementation |
| matplotlib / seaborn | Data visualization |
| Jupyter Notebook | Interactive development environment |

---

## References

[1] Cordeiro, A. M. R., & Alcoforado, L. (2018). *Education and Development: Contributions to the Changes of Democratic Portugal*. OpenEdition Journals.

[2] Dubow, E. F., Boxer, P., & Huesmann, L. R. (2009). Long-term Effects of Parents' Education on Children's Educational and Occupational Success. *Merrill-Palmer Quarterly*, 55(3), 224–249.

[3] Gandhi, A., Kumar, A., Konkimalla, A., & Desai, S. (2021). *Deeper Look into Academic Performance of Portuguese Students*. Carnegie Mellon University.

[4] Wang, Y. (2024). Impact of Love and Romantic Relationships on Adolescent Psychology and Their School Performance. *Proceedings of the 3rd International Conference on Interdisciplinary Humanities and Communication Studies*.

[5] Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. University of Minho.
