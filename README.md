# Compliance Radar — Machine Learning 2025/2026  
LUISS Guido Carli — Bachelor in Management & AI  

## Team Members  
- Petra Babic (Team Captain)  
- Boyan Aleksandrov  
- Ayan Alybay  
- Koray Aydin  

# Project Overview  
**Compliance Radar** is a machine-learning project that identifies potential non-compliance signals across organisational departments.  
Using the SQLite dataset `data/org_compliance_data.db`, we label departments as **high risk (1)** if they appear in the `high_risk_departments` table, and **low/normal risk (0)** otherwise.  

The goal is not only prediction, but also **interpretability**: we compare transparent baseline models against stronger ensemble models, evaluate trade-offs (false alarms vs missed risks), and provide insights that can support compliance monitoring and governance decisions.

# Repository Structure  
- `/images/` — exported figures (confusion matrices, feature importance, ROC curves)  
- `main.ipynb` — full notebook (end-to-end workflow)  
- `README.md` — project documentation  
- `environment.yml` — environment dependencies  
- `data/org_compliance_data.db` — dataset (not pushed if large)  

# Notebook Structure (main.ipynb)  
The notebook follows the required course structure:  
- Data loading  
- Exploratory Data Analysis (EDA)  
- Preprocessing & feature engineering  
- Model training & cross-validation  
- Evaluation & interpretability  
- Conclusions & insights  

# Data Loading  
We load the SQLite database from the `data/` folder and inspect available tables with SQLAlchemy.  
Main tables used:  
- `departments` — department-level features  
- `high_risk_departments` — reference table used to define the target label  

## Target Construction (high_risk)  
We create the binary target as:  
- `high_risk = 1` if `dept_id` is found in `high_risk_departments`  
- `high_risk = 0` otherwise  

The dataset contains **709** departments in total, with **217 (~30.6%)** labeled high risk and **492 (~69.4%)** low/normal risk.  
Because missing a risky department is costly in compliance contexts, we focus on **precision, recall, and F1-score**, not accuracy alone.

# Exploratory Data Analysis  
EDA is used to understand:  
- class balance (moderate imbalance)  
- missingness patterns  
- relationships between key variables/categories and the target  

## Missingness Handling (Feature Removal)  
A missingness scan shows many variables with high missing values.  
To reduce noise and avoid heavy imputation, we drop features with **missingness > 40%**, leaving **5 core predictors** used in modeling.

## Key Patterns Observed  
- Audit-related numeric variables show stable distributions and are strongly linked to the compliance outcome.  
- `dept_category` shows clear risk differences across department functions, supporting its inclusion as an explanatory feature.

# Preprocessing & Feature Handling  
We use a unified scikit-learn preprocessing pipeline across models to ensure consistency and prevent leakage.

## Preprocessing Pipeline  
- **Numerical features:** median imputation + standard scaling  
- **Categorical features:** most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)  

All preprocessing is implemented inside a `Pipeline` so it is fitted only on the training data.

## Train/Test Split  
We use an **80/20 stratified split** to preserve the high-risk proportion in both sets:  
- training: 567 rows  
- test: 142 rows  

# Models  
We evaluate three classifiers to balance interpretability and predictive power:

## Logistic Regression (Baseline + Tuned)  
Logistic Regression is the baseline model due to interpretability and stable probability outputs (usable as risk scores).  
We tune it with **GridSearchCV** (5-fold stratified CV), optimising for **F1-score** by varying the regularisation strength `C`.

## Random Forest (Tuned + Threshold Optimisation)  
Random Forest is used to capture non-linear relationships and feature interactions.  
We tune hyperparameters using **GridSearchCV** (5-fold stratified CV), and then perform **threshold optimisation** using cross-validated predicted probabilities to select the decision threshold that maximises F1-score.  
This aligns the final decision rule with compliance priorities (reduce missed high-risk departments while controlling false alarms).

## XGBoost (Baseline + Tuned)  
XGBoost is included because it typically performs strongly on structured/tabular datasets and can model complex interactions efficiently.  
To address class imbalance, we compute and apply `scale_pos_weight = (#neg / #pos)` during training.  
We train:  
- a baseline XGBoost model  
- a tuned model via **GridSearchCV** (5-fold CV), optimising F1-score over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

# Evaluation Metrics  
All models are evaluated on the held-out test set using:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **ROC-AUC** (for probability ranking quality)  

We also analyse:  
- confusion matrices (error types: false positives vs false negatives)  
- ROC curves (threshold-independent comparison)  
- Random Forest feature importance (interpretability)

# Results Summary  
Across models, accuracy is generally high, but the main differences appear in **precision, recall, and F1-score**, which matter most for compliance monitoring.

------------------------------------------------------------------------------------
model               | Accuracy | Precision | Recall | F1-score
-----------------------------------------------------------------------------------
Logistic Regression |  0.87    |   0.80    |  0.77  |  0.79
------------------------------------------------------------------------------------
Random Forest       |  0.91    |   0.90    |  0.81  |  0.85
------------------------------------------------------------------------------------
XGBoost             |  0.92    |   0.94    |  0.79  |  0.86
------------------------------------------------------------------------------------
^all metrics above are for given model's tuned versions 

## Logistic Regression  
Provides a clear and interpretable benchmark, but misses a meaningful portion of high-risk departments (false negatives), motivating ensemble methods.

## Random Forest  
Improves performance over Logistic Regression by capturing non-linear patterns.  
Threshold optimisation makes the decision policy explicit and aligned with compliance objectives.

## XGBoost  
The tuned XGBoost model achieves the strongest overall performance (highest precision/F1 and strong ROC-AUC).  
It reduces false positives while maintaining strong recall, making it especially practical when compliance teams want fewer incorrect risk flags without losing too much sensitivity to true risk.

# Interpretability  
Interpretability is treated as a core requirement for compliance use cases.

## Feature Importance (Random Forest)  
Feature importance highlights that **audit and compliance score variables** are dominant predictors of high-risk classification.  
This supports the practical interpretation that past audit outcomes and compliance performance are strong early warning signals for future compliance risk.

# Conclusions  
The project demonstrates that machine learning can support compliance teams by flagging high-risk departments and explaining what drives risk.

## Main Takeaways  
- Logistic Regression is valuable for transparency, but limited in predictive strength.  
- Random Forest improves detection and balances precision/recall well.  
- Tuned XGBoost provides the best overall performance on this dataset, making it the most robust choice for deployment-oriented compliance monitoring.

## Ethical & Practical Notes  
- Predictions should support human decision-making, not replace it.  
- Outputs must remain explainable to auditors and decision-makers.  
- Regular monitoring and retraining are required to avoid drift and unintended bias.

## Future Work  
- Add time-based features to track compliance evolution.  
- Use clustering to detect hidden department profiles.  
- Incorporate text fields from audit reports for richer context.  
- Deploy with monitoring (performance + drift detection) for real-world use.

# Course Requirements  
This repository follows the mandatory Machine Learning 2025/2026 group project structure and includes:  
data loading, EDA, preprocessing, multiple models with tuning, evaluation with proper metrics, and interpretability-driven analysis.
