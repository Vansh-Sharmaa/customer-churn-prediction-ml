# Customer Churn Prediction: Beyond Accuracy

This project demonstrates a real-world approach to predicting customer churn. Instead of simply building a machine learning model and chasing high accuracy scores, this project focuses on solving the actual business problem by emphasizing interpretability, proper handling of imbalanced data, and choosing the right evaluation metrics.

## Key Focus Areas

1. **Identifying Key Churn Drivers**: Extracting feature weights from the model to understand *why* customers churn based on factors like Contract Type, Tenure, and Monthly Charges.
2. **Handling Class Imbalance**: In the real world, churned customers are usually a monetary minority. The project uses built-in balanced class weights to ensure the minority class is learned effectively.
3. **Evaluating Beyond Accuracy**: Relying heavily on the **F1-score**, which provides a much more robust evaluation measurement for imbalanced datasets than standard accuracy.
4. **Interpreting Model Insights**: Using a highly-interpretable **Logistic Regression** model so findings can be effortlessly communicated to non-technical stakeholders for targeted business action.

## Project Structure

* `churn_model.py`: The complete end-to-end Python script that generates a synthetic imbalanced dataset, processes it, trains a Logistic Regression model, evaluates with F1-score, and outputs clear business insights.

## How to Run

1. Ensure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Run the script:
```bash
python churn_prediction_model.py
```

## Why Logistic Regression?
While Black-box models (like Random Forests or Gradient Boosted Trees) might yield incrementally higher performance metrics, business stakeholders cannot act on "black box" decisions. Logistic Regression provides explicit feature coefficients, telling us exactly how much a specific attribute (like moving a customer from a Month-to-Month to a 1-Year contract) impacts their likelihood of churning.

