# ibm-hr-attrition-prediction
A machine learning project using Python and Logistic Regression to explore and predict employee attrition.

Overall Model Performance
Accuracy (84%): At first glance, an 84% accuracy rate seems excellent. However, in this case, accuracy is highly misleading due to class imbalance. Since 84% of the employees in your test set stayed (No Attrition), a simple model that guesses "No Attrition" for everyone would also achieve 84% accuracy without learning any patterns.

Key Evaluation Metrics Breakdown
Precision (66.67%): Out of all employees the model predicted would leave, 66.67% actually did. This is moderately strong, meaning when the model raises a red flag, it is more likely to be correct than not.
Recall / Sensitivity (4.26%): Out of all employees who actually left (47 people in the test set), the model only successfully caught 2 of them (4.26%). It missed 95.74% of the employees who resigned.
F1-Score (0.0800): Since the recall is extremely low, the F1-score (which balances precision and recall) is very poor. This indicates the model is struggling to identify attrition.
ROC-AUC Score (0.5687): A score of 0.50 represents a random guess. A score of 0.57 means your current features (such as Age, DailyRate, and DistanceFromHome) have very weak predictive power to distinguish between employees who stay and those who leave under standard logistic regression settings.

Why did this happen and how can we fix it?
The Threshold Problem: Logistic regression predicts probabilities. By default, it classifies anyone with a probability above 0.50 as "Yes" (Attrition). Because "Yes" cases are rare in the data, the model's predicted probabilities rarely cross 0.50.
Actionable Next Steps: We can drastically improve the model's sensitivity (recall) by either:
Balancing the class weights using class_weight='balanced' inside the model configuration.
Adjusting the probability classification threshold (e.g., classifying a "Yes" at 0.20 instead of 0.50).
