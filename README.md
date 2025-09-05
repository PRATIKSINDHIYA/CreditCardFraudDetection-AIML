
# Credit Card Fraud Detection (AI&ML Project)

## Abstract
Credit card fraud is a significant problem, causing billions of dollars in losses each year. Machine learning can detect fraudulent transactions by identifying patterns indicative of fraud. This project develops a machine learning model to detect credit card fraud using historical transaction data. The model is trained on past transactions and evaluated on unseen data for performance assessment.

**Keywords:** Credit Card Fraud Detection, Fraudulent Transactions, K-Nearest Neighbors, Support Vector Machine, Logistic Regression, Decision Tree.

---

## Overview
With the increasing use of credit cards, security has become critical. Reports of credit card fraud are rising worldwide. This project applies machine learning techniques to detect fraudulent credit card transactions efficiently.

---

## Project Goals
- Detect fraudulent credit card transactions to prevent unauthorized charges.  
- Implement multiple machine learning algorithms and compare their performance.  
- Provide visualizations and metrics for analysis.  
- Explore techniques from previous studies for detecting fraud in financial datasets.

---

## Dataset
The dataset was obtained from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  

- **Rows:** 284,808  
- **Attributes:** 31  
- **Details:**  
  - 28 numeric features (PCA-transformed)  
  - Time: elapsed seconds between first and other transactions  
  - Amount: transaction amount  
  - Class: 1 = fraud, 0 = non-fraud  

---

## Algorithms Used
1. **K-Nearest Neighbors (KNN)**  
2. **Logistic Regression (LR)**  
3. **Support Vector Machine (SVM)**  
4. **Decision Tree (DT)**  

---

## Methodology
1. Data preprocessing using **Pandas** and **NumPy**  
2. Handling missing values and scaling features  
3. Train-test split for model evaluation  
4. Model training with KNN, LR, SVM, and Decision Tree  
5. Evaluation using accuracy, precision, recall, and F1-score  
6. Comparison of model performance to identify the best model  

---

## Future Work
- Apply the model to larger or more diverse datasets  
- Experiment with different preprocessing or train-test splits  
- Integrate additional data sources (e.g., location data) to enhance fraud detection  

---

## Conclusion
The project successfully identified the most effective machine learning models for detecting credit card fraud. KNN and Decision Tree achieved the highest accuracy, helping improve transaction security and customer satisfaction.

---

## Project Structure
```

credit-card-fraud-detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud\_detection.ipynb
├── src/
│   ├── data\_preprocessing.py
│   ├── model\_training.py
│   └── evaluation.py
├── requirements.txt
└── README.md

````

---

## Requirements
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/PRATIKSINDHIYA/
cd CreditCardFraudDetection-AIML
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---
## Contact
**Pratik Sindhiya**  
Email: [pratiksindhiya3@gmail.com](mailto:pratiksindhiya3@gmail.com)  
Feel free to reach out for any queries regarding this project.
