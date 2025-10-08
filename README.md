# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company. By identifying customers at high risk of churning, the company can take proactive steps to retain them, thereby preserving revenue and reducing acquisition costs.

## Author

* **Bruno Jardim**

## Table of Contents
- [Business Problem](#business-problem)
- [Data](#data)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings from EDA](#key-findings-from-eda)
- [Modeling Results](#modeling-results)
- [Getting Started](#getting-started)
- [Tools and Libraries](#tools-and-libraries)
- [Author](#author)

---

## Business Problem

The primary goal is to develop a predictive model that identifies customers likely to churn. This allows the business to implement targeted retention strategies, such as discounts, service upgrades, or special offers, to reduce customer attrition and maximize the return on investment (ROI) of these campaigns. The key business metrics for success include the number of customers retained, the amount of revenue preserved, and the overall ROI of the retention efforts.

---

## Data

The dataset used is the "WA_Fn-UseC_-Telco-Customer-Churn.csv" file, which contains 7,043 customer records and 21 variables. These variables include:

* **Demographic Information:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
* **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
* **Target Variable:** `Churn` (Yes/No), indicating whether the customer has churned.

---

## Project Structure

The project is organized into a series of Jupyter notebooks that follow a structured data science workflow:

* `00_DSW_Problema_de_Negocio_Telco.ipynb`: Defines the business problem, objectives, and success metrics.
* `01_DSW_Entendimento_Dos_Dados.ipynb`: Involves initial data loading, quality checks, and exploratory data analysis (EDA) to understand the customer profiles and churn patterns.
* `02_DSW_DataPrep.ipynb`: Focuses on cleaning and preparing the data for modeling. This includes handling missing values, encoding categorical variables, and scaling numerical features.
* `03_DSW_Feature_Selection.ipynb`: Selects the most relevant features for the predictive models to improve performance and interpretability.
* `04_Machine_Learning.ipynb`: Implements and evaluates several machine learning models to predict churn, followed by a comparison of their performance.

---

## Methodology

The project follows these key steps:

1.  **Exploratory Data Analysis (EDA):** The initial analysis of the dataset revealed several key insights into churn behavior. For example, customers with month-to-month contracts, fiber optic internet service, and no tech support have a higher tendency to churn.
2.  **Data Preparation:** The data was preprocessed to make it suitable for machine learning models. This involved converting the target variable `Churn` to a binary format, handling missing values in `TotalCharges`, and transforming other variables as needed.
3.  **Feature Selection:** A feature selection process was employed to identify the most predictive variables. Techniques like analyzing feature importance from a Gradient Boosting model were used to reduce the number of features to 34, which helps in creating a more robust and efficient model.
4.  **Machine Learning Models:** Several classification models were trained and evaluated, including:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting
    * Support Vector Machine (SVM).

    The models were trained using `GridSearchCV` to find the best hyperparameters, and their performance was evaluated based on metrics such as **AUC**, **Precision**, and **Recall**.

---

## Key Findings from EDA

The exploratory data analysis provided several important insights:
* **Churn Rate:** The overall churn rate is approximately 26.5%.
* **Contract Type:** Customers with **month-to-month contracts** are significantly more likely to churn compared to those with one or two-year contracts.
* **Internet Service:** Customers with **fiber optic** internet service show a higher churn rate.
* **Additional Services:** Lack of services like **online security** and **tech support** is correlated with higher churn.

---

## Modeling Results

The Gradient Boosting model demonstrated the best overall performance, achieving the highest AUC score of **0.846**. While other models like Random Forest and Logistic Regression also performed well, Gradient Boosting provided a slightly better balance of precision and recall, making it the most suitable choice for this business problem.

Here's a summary of the model performance on the test set:

| Model | AUC | Precision | Recall | Gini |
| :--- | :--- | :--- | :--- | :--- |
| GradientBoosting | 0.8457 | 0.6608 | 0.5000 | 0.6914 |
| RandomForest | 0.8430 | 0.5395 | 0.7487 | 0.6860 |
| LogisticRegression | 0.8410 | 0.5069 | 0.7834 | 0.6819 |
| SVC | 0.8100 | 0.0000 | 0.0000 | 0.6199 |


---

## Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/telco-churn-prediction.git](https://github.com/your-username/telco-churn-prediction.git)
    cd telco-churn-prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter notebooks:**
    Start Jupyter Lab or Jupyter Notebook and run the notebooks in sequential order (`00` to `04`).
    ```bash
    jupyter lab
    ```

---

## Tools and Libraries

* **Python 3**
* **Pandas** for data manipulation and analysis.
* **Scikit-learn** for machine learning models and preprocessing.
* **Matplotlib** and **Seaborn** for data visualization.
* **Jupyter Notebook** for interactive development.
