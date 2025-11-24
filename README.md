# Machine-Learning-2025-2026---Group-Project
# Compliance Radar — Machine Learning 2025/2026  
LUISS Guido Carli — Bachelor in Management & AI  

## Team Members  
- Petra Babic (Team Captain)  
- Boyan Aleksandrov
- Ayan Alybay
- Koray Aydin

---

# 1. Introduction  

The **Compliance Radar** project aims to identify potential non-compliance signals and ethical risk patterns within organisations.  
Using the dataset `org_compliance_data.db`, which includes operational metrics, audit indicators, engagement variables, financial scores, and departmental activity logs, our goal is to build a **data-driven framework** that:

- Detects unusual or potentially risky behaviours,  
- Highlights the features most correlated with non-compliance,  
- Provides interpretable and actionable insights for internal governance,  
- Goes beyond prediction to support ethical reasoning and accountability.

This project follows the Machine Learning course structure and integrates both **quantitative modelling** and **interpretability-focused analysis** to strengthen organisational integrity.

---

# 2. Methods  

## 2.1 Workflow Overview  
Our ML pipeline includes:

1. **Loading & exploring the dataset**  
2. **EDA and visual inspection**  
3. **Data cleaning & preprocessing**  
4. **Feature engineering**  
5. **Training multiple models** (minimum 3 required)  
6. **Cross-validation & hyperparameter tuning**  
7. **Model evaluation using proper metrics**  
8. **Interpretability analysis (feature importance, SHAP)**  
9. **Data-driven recommendations for risk mitigation**

## 2.2 Algorithms Used  
To comply with project requirements, we test at least three algorithms:

- **Logistic Regression**  
  - Baseline model, highly interpretable and useful for identifying directional influence of features.

- **Random Forest Classifier**  
  - Handles non-linear patterns and interactions effectively.

- **XGBoost Classifier**  
  - Powerful tree-based method that often performs best on tabular datasets with mixed feature types.

(Additional optional models may be included depending on data behaviour.)

## 2.3 Preprocessing Steps  
- Handling missing values  
- Detecting & removing outliers  
- Encoding categorical variables  
- Feature scaling (when needed)  
- Checking class imbalance  
- Optional: SMOTE for balancing minority classes  

## 2.4 Environment  
The project environment can be recreated using:
pip install -f environment.yml

## 2.5 System Flowchart  
      ┌────────────────────┐
      │   Load Database    │
      └─────────┬──────────┘
                ▼
    ┌────────────────────────────┐
    │ Exploratory Data Analysis  │
    └───────────┬──────────────-─┘
                ▼
     ┌───────────────────────────┐
     │ Preprocessing & Cleaning  │
     └────────────┬──────────────┘
                  ▼
     ┌───────────────────────┐
     │ Feature Engineering   │
     └─────────┬─────────────┘
               ▼
    ┌───────────────────────────────────┐
    │ Model Training & Cross-Validation │
    └──────────────┬────────────────────┘
                   ▼
    ┌───────────────────────────-───-┐
    │ Evaluation (Accuracy, F1, AUC) │
    └────────────┬──────────────-──-─┘
             ▼
    ┌─────────────────────────────────────-┐
    │ Interpretability (Feature Importance │
    │             + SHAP)                  │
    └────────────────┬───────────────────-─┘
                     ▼
    ┌──────────────────────────────────┐
    │ Insights & Compliance Reasoning  │
    └──────────────────────────────────┘
---

# 3. Experimental Design  

## 3.1 Purpose of Experiments  
Our experiments aim to:

- Compare predictive models for detecting compliance risk,  
- Understand which variables contribute the most to risk,  
- Validate model robustness through cross-validation,  
- Balance accuracy with interpretability (critical for compliance work).

## 3.2 Baselines  
We compare all models against:

- **Majority class classifier** (predict most frequent class)  
- **Logistic Regression baseline**

## 3.3 Evaluation Metrics  

Following course guidelines (accuracy, precision, recall, F1, ROC-AUC), we evaluate models using:

- **Accuracy** — general performance  
- **Precision** — how many detected risks were real  
- **Recall** — how many true risks we successfully detected  
- **F1-score** — balance between precision & recall  
- **ROC-AUC** — model ability across thresholds  
- **Confusion Matrix** — error pattern understanding  

These metrics ensure fairness and reliability, especially if classes are imbalanced.

---

# 4. Results  

## 4.1 Summary of Findings  
(Results will be added after running models in `main.ipynb`.)

The results section will include:

- Best-performing model,  
- Key feature importance rankings,  
- Visualisations (ROC curve, confusion matrix, SHAP plots),  
- Insights on departmental risk patterns and behavioural factors.

## 4.2 Example Table (to be filled when results are obtained)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|----|-----|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

All final figures will be saved in:  
---

# 5. Conclusions  

## 5.1 Takeaways  
The Compliance Radar framework provides a structured, interpretable, and evidence-based way to assess organisational risk.  
By analysing behavioural, operational, and financial patterns, it supports compliance teams in identifying vulnerabilities and responding proactively.

## 5.2 Limitations & Future Work  
- Temporal modelling would improve trend detection.  
- Adding clustering could reveal hidden departmental archetypes.  
- Incorporating textual reports or qualitative feedback would deepen ethical insights.  
- Deploying the model with a monitoring pipeline would support real-time compliance tracking.

---

# Repository Structure
/images/                      # Figures used in README
main.ipynb                   # Full project notebook
README.md                    # Project documentation
environment.yml              # Environment dependencies
data/org_compliance_data.db  # Dataset (not pushed if large)


Project follows all mandatory requirements from the Machine Learning course (2025/2026).

           
