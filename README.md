# Machine-Learning-2025-2026---Group-Project
# Compliance Radar — Machine Learning 2025/2026  
LUISS Guido Carli — Bachelor in Management & AI  

## Team Members  
- Petra Babic (Team Captain)  
- Boyan Aleksandrov
- Koray Aydin

---

# 1. Introduction  

This project, called Compliance Radar, looks at how we can use machine learning to spot possible compliance risks inside organisations. Our dataset includes 709 departments and a range of operational, behavioural, and audit-related indicators.
The idea is to build a system that helps identify which departments might require extra attention and why.

To do this, we follow the main steps discussed in class:
	1.	loading and inspecting the data,
	2.	exploring patterns and distributions,
	3.	preparing the variables for modelling,
	4.	training several machine learning models,
	5.	comparing their performance,
	6.	interpreting the results with tools such as feature importance and SHAP.

The goal is not only prediction. We also want to understand why certain departments appear riskier and how the organisation could react. This combination of performance and interpretability is essential for topics like compliance, where decisions must be transparent and defensible.

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

We included XGBoost because it usually performs well on tabular datasets and can capture more complex interactions than simpler models.
The target variable was created by marking departments with a final compliance score below 70 as “high risk”.

After splitting the data (80% train, 20% test), we trained a baseline XGBoost model with default parameters.
Its test performance was already strong:
	•	Accuracy: around 0.94
	•	Precision: ~0.89
	•	Recall: very high at ~0.98
	•	F1-score: ~0.93
	•	AUC: ~0.96

The high recall is especially important here. In compliance, missing a risky department is more costly than flagging one extra.

We then performed a small grid search over three hyperparameters:
n_estimators, max_depth, and learning_rate.
The best combination was:
	•	n_estimators = 200
	•	max_depth = 3
	•	learning_rate = 0.01

The tuned model produced similar results to the baseline. Accuracy and precision changed slightly, but recall stayed almost the same and remained very high.
This shows that XGBoost is stable on this dataset and consistently identifies risky departments.

# 5. Conclusions  
The project shows that machine learning can help highlight which departments inside an organisation might face higher compliance risks.
By bringing together EDA, model development, and interpretability tools, we managed to form a clearer picture of the factors connected with lower compliance performance.

5.1 Main Observations
	•	XGBoost performed the best overall when balancing accuracy and recall.
	•	The model was particularly good at identifying nearly all high-risk cases, which is crucial for compliance teams.
	•	SHAP results give a clearer sense of which variables push a department’s risk level up or down.
Common influential factors include audit scores, operational risk exposure, reporting gaps, and resource availability.

5.2 Ethical Points

A system like this must remain understandable.
Risk predictions should support human decisions, not replace them.
Any model also needs regular monitoring and retraining so that it does not drift or become biased over time.

5.3 Future Ideas
	•	Adding time-based data would help track how departments evolve across years.
	•	Clustering methods could reveal hidden patterns or groups among departments.
	•	Text data from audit reports could add more context.
	•	Deploying the model with a monitoring component could help organisations track changes more dynamically.

# Repository Structure
/images/                      # Figures used in README
main.ipynb                   # Full project notebook
README.md                    # Project documentation
environment.yml              # Environment dependencies
data/org_compliance_data.db  # Dataset (not pushed if large)


Project follows all mandatory requirements from the Machine Learning course (2025/2026).

           
