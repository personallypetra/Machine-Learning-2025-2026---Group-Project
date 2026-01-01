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

This project develops a data-driven framework for identifying potential compliance risks across organisationaal departments. Analysis combines interpretable baseline models with more advanced ensamble methods to balance transparency, predictive performance, and practical relevance for compliance monnitoring. 

## Baseline Model: Logistic Regression 
Logistic Regression is used as baseline classifier due to its interpretability and stability. Model provides coefficient based insights into how individual features increase or decrease the probability of a department being classified as high-risk. This transparency is particularly important in compliance settings, where model decisions must be explainaable to auitors and decision-makers. 

Additionally, Logistic Regression outputs probabilities that can be interpreted as **risk scores**, making it a standard and appropriate benchmark. 

### Ensamble Models: Random Forest 
Random Forest is employed to capture non-linear relationships and feature interactions present in organisational and operational data. By aggregating multiple decision trees traiined on bootstrappped samples, Random Forest improves robustness and reduces variance. 

Both a default Random Forest model and tuned version are evaluated to assess how hyperparameter optimisation improves compliance risk detection. 

##Preprocessing and Feature Handling 
The dataset contains numerical and categorical features, as well as missing values. A unified preprocessing pipeline is applied consistently across all models: 
• Numerical features are imputed using the median and scaled using standardisation
• Categorical features are imputed using the most frequent category and encoded using one-hot encoding

This pipeline ensures consistency, prevent data leakage, and allows fair comparison across models. 

## Evaluation Metrics 
Model Performance is evaluated using **accuracy**, **precision**, **recall**, and **F1-score**, with additional analysis using confusion matrices and ROC curves. Given the class imbalance and high cost of missclassifying high-risk departments, particular emphasis is placed on **precision**, **recall** and **F1-score**, rather than accuracy alone. 

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

## Purpose of Experiments

The experiments aim to: 
• Compare baseline and tuned models for compliance risk detection, 
• Assess the impact of hyperparameter tuning, 
• Evaluate trade-offs between interpretability and predictive performance, 
• Identify models best aligned with compliance objectives 

## Train-Test Strategy 

The dataset is split into training and test sets using a stratified split to preserve the proportion of high-risk and low-risk departments. All preprocessing and tuning steps are applied exclusively to the training set. 

## Hyperparameter Tuning


Hyperparameter optimization is performed using **cross-validation**:
• Logistic Regression is tuned using **GridSearchCV**, varying the regularisation strength. 
• Random Forest is tuned using **GridSearchCV**, exploring:
  - numer of trees,
  - maximum tree depth,
  - minimum samples per split and leaf,
  - feature subsampling strategy
Five-fold stratified cross-validation is used, with **F1-score** as the optimisation metric to balance precision and recall.
---
# 4. Results 

## Model Performance Comparison 

The final performance of Logistic Regression (baseline and tuned) and Random Forest (baseline and tuned) is summarised using accuracy, precision, recall, and F1-score. 
While accuracy remains relatively similar across all models, substantial differences emerge when examining precision, recall, and F1-score. 

The tuned Random Forest model achieves the strongest overall performance, particularly in terms of F1-score, indicating a superior balance between identifying high-risk departments and limiting false alarms. Logistic Regression provides a strong and interpretable baseline, and tuning improves its performance but it does not fully match the effectiveness of Random Forest. 

## Confusion Matrix Analysis 

Confusion matriz analysis reveals important differences in error patterns. Logistic Regression produces a higher number of false positives and false negatives, leading to unnecessary compliance investigations and missed high-risk departments.

The tuned Random Forest model reduces false positives from 8 to 4 and false negatives from 10 to 8, demonstrating improved risk detection while maintaining operational efficiency. 

## ROC Curve Analysis 

ROC curve comparisons further confirm that Random Forest exhibits stronger discriminative ability across classification tresholds compared to Logistic Regression. The higher are under the curve (AUC) indicates more reliable ranking of departments by risk level.

## Final Model Selection

Although Logistic Regression offers interpretability and serves as a valuable baseline, the tuned Random Forest model demonstrates superior performance on the metrics most relevant to compliance monitoring - particularly precision, recall, and F1-score. 
By reducing both missed risks and unwarranted audits, Random Forest aligns most closely with organisational compliance objectives and is therefore selected as the final model. 


