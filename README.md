Here is the same text with all headings starting with # so you can copy-paste directly into your report.

⸻

# 2. Methods

This project develops a data-driven framework for identifying potential compliance risks across organisational departments. The analysis combines interpretable baseline models with advanced ensemble methods to balance transparency, predictive performance, and practical relevance for compliance monitoring.

## Baseline Model: Logistic Regression

Logistic Regression is used as the baseline classifier due to its interpretability and stability. The model provides coefficient-based insights into how individual features increase or decrease the probability of a department being classified as high-risk. This transparency is particularly important in compliance environments, where model decisions must be explainable to auditors and governance teams.

In addition, Logistic Regression outputs probabilities that can be interpreted as continuous risk scores, making it a suitable benchmark for comparison with more complex models.

## Ensemble Models: Random Forest and XGBoost

Random Forest is employed to capture non-linear relationships and feature interactions in organisational and operational data. By aggregating multiple decision trees trained on bootstrapped samples, Random Forest improves robustness and reduces variance compared to single-tree models.

In addition, the XGBoost (Extreme Gradient Boosting) algorithm is implemented to further enhance predictive performance. XGBoost builds trees sequentially, where each new tree corrects the errors of the previous ones. This boosting mechanism enables the model to focus on difficult-to-classify observations and often leads to superior performance on structured tabular datasets.

Both Random Forest and XGBoost are evaluated in default and tuned configurations to assess the impact of hyperparameter optimisation on compliance risk detection.

## Preprocessing and Feature Handling

The dataset contains numerical and categorical features as well as missing values. A unified preprocessing pipeline is applied consistently across all models:

• Numerical features are imputed using the median and scaled using standardisation
• Categorical features are imputed using the most frequent category and encoded using one-hot encoding

This pipeline prevents data leakage, ensures consistency, and allows fair comparison across models.

## Evaluation Metrics

Model performance is evaluated using accuracy, precision, recall, and F1-score, supported by confusion matrices and ROC curves. Given the class imbalance and the high cost of misclassifying high-risk departments, particular emphasis is placed on precision, recall, and F1-score, rather than accuracy alone.

# 3. Experimental Design

## Purpose of Experiments

The experiments aim to:

• Compare baseline and ensemble models for compliance risk detection
• Assess the impact of hyperparameter tuning
• Evaluate trade-offs between interpretability and predictive performance
• Identify the model best aligned with compliance monitoring objectives

## Train-Test Strategy

The dataset is split into training and test sets using a stratified split to preserve the proportion of high-risk and low-risk departments. All preprocessing, training, and tuning steps are applied exclusively to the training set.

## Hyperparameter Tuning

Hyperparameter optimisation is performed using GridSearchCV with five-fold stratified cross-validation and F1-score as the optimisation metric.

• Logistic Regression: tuning of the regularisation strength
• Random Forest: tuning of number of trees, maximum depth, minimum samples per split and leaf, and feature subsampling strategy
• XGBoost: tuning of learning rate, maximum depth, number of estimators, and subsampling parameters

# 4. Results

## Model Performance Comparison

The final performance of Logistic Regression, Random Forest, and XGBoost (default and tuned) is summarised using accuracy, precision, recall, and F1-score. While accuracy values remain relatively similar across all models, notable differences emerge in precision, recall, and F1-score.

The tuned Random Forest model achieves the strongest overall performance, closely followed by the tuned XGBoost model. Logistic Regression provides a solid and interpretable baseline, but even after tuning it does not fully match the effectiveness of the ensemble approaches.

## Confusion Matrix Analysis

Confusion matrix analysis reveals significant differences in error patterns. Logistic Regression produces a higher number of false positives and false negatives, leading to unnecessary compliance investigations and missed high-risk departments.

The tuned Random Forest model reduces false positives from 8 to 4 and false negatives from 10 to 8, demonstrating improved detection accuracy and operational efficiency. XGBoost shows a similar improvement trend, but with slightly higher misclassification rates than Random Forest.

## ROC Curve Analysis

ROC curve comparisons confirm that Random Forest and XGBoost exhibit stronger discriminative ability across classification thresholds compared to Logistic Regression. The tuned Random Forest model achieves the highest AUC, indicating the most reliable ranking of departments by compliance risk.

# Final Model Selection

Although Logistic Regression remains valuable for interpretability, the tuned Random Forest model demonstrates superior performance on the metrics most relevant to compliance monitoring – particularly precision, recall, and F1-score.

By reducing both missed compliance risks and unnecessary investigations, Random Forest aligns most closely with organisational compliance objectives and is therefore selected as the final model.
