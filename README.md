Prediction: Classification Models and Streamlit App

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP | **Assignment 2**

**Live app:** https://credit-card-default-ml-classifier.streamlit.app/

---

## a. Problem statement

Credit card issuers lose money when customers fail to pay their bills. The goal of this project is to predict, from a customer's profile and their last six months of repayment history, whether the customer will **default on the next month's payment** (binary classification: `0` = no default, `1` = default).

Five classification models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine) are trained on the same dataset and compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The trained models are deployed in an interactive Streamlit web app, where a test CSV can be uploaded and evaluated without retraining.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Default of Credit Card Clients |
| Source | UCI Machine Learning Repository (also on Kaggle as "Default of Credit Card Clients Dataset") |
| Problem type | Binary classification |
| Instances | 30,000 customers (Taiwan, 2005). Minimum required: 500 |
| Features | 23 input features, after dropping the `ID` column. Minimum required: 12 |
| Target | `default` (originally `default.payment.next.month`): 0 = no default, 1 = default |
| Class balance | 77.9% no default, 22.1% default (imbalanced) |
| Missing values | None |

**Feature groups**
- **Credit and demographics (5):** `LIMIT_BAL` (credit limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Repayment status (6):** `PAY_0`, `PAY_2` to `PAY_6` (repayment status for the last six months, from September back to April 2005)
- **Bill amounts (6):** `BILL_AMT1` to `BILL_AMT6`
- **Previous payment amounts (6):** `PAY_AMT1` to `PAY_AMT6`

**Preprocessing**
- Dropped the `ID` column and renamed the target to `default`.
- `EDUCATION` contains undocumented codes (0, 5, 6), which were merged into category 4 ("other"). `MARRIAGE` code 0 was merged into category 3 ("other").
- Stratified 80/20 train/test split (`random_state=42`), giving 24,000 training rows and 6,000 test rows. The test split is saved as `test_data.csv`.
- `StandardScaler` is part of the pipeline for Logistic Regression, KNN, Naive Bayes and SVM, so each saved model includes its own scaling. The Decision Tree does not need scaling.

## c. GitHub repository link

https://github.com/Nataraj-TR/credit-card-default-ml-classifier

**Repository structure**
```
├── app.py                  # Streamlit application
├── train_models.py         # Data preparation, training, evaluation, model saving
├── requirements.txt        # Dependencies for deployment
├── README.md
├── test_data.csv           # 6,000-row held-out test set (features + target)
├── metrics_results.csv     # Evaluation results of all 5 models
└── model/                  # Saved trained pipelines (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 6,000-row test set.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, with StandardScaler |
| Decision Tree | `max_depth=5` |
| KNN | `n_neighbors=15`, with StandardScaler |
| Naive Bayes | Gaussian, with StandardScaler |
| Support Vector Machine | RBF kernel, `C=1.0`, with StandardScaler (AUC computed from `decision_function`) |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.8083 | 0.7079 | 0.6903 | 0.2419 | 0.3583 | 0.3276 |
| Decision Tree | 0.8172 | 0.7418 | 0.6615 | 0.3549 | 0.4620 | 0.3893 |
| KNN | 0.8062 | 0.7319 | 0.6152 | 0.3301 | 0.4296 | 0.3483 |
| Naive Bayes | 0.7518 | 0.7248 | 0.4504 | 0.5539 | 0.4968 | 0.3376 |
| Support Vector Machine | 0.8162 | 0.7186 | 0.6662 | 0.3384 | 0.4488 | 0.3814 |

Precision, Recall and F1 are reported for the positive class (default = 1).
The live Streamlit app runs a newer scikit-learn version than the training environment, so its KNN row differs very slightly (Accuracy 0.8060, Precision 0.6146, Recall 0.3293, F1 0.4289, MCC 0.3476; AUC is unchanged). This is a difference of one or two test rows caused by the library version, and the other four models match exactly.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.69) but the lowest recall (0.24), F1 (0.36) and MCC (0.33). It is very conservative: when it flags a customer as a defaulter it is usually right, but it misses about three quarters of the actual defaulters. A linear decision boundary is probably too simple for the non-linear relation between repayment history and default. |
| Decision Tree | Best accuracy (0.817), AUC (0.742) and MCC (0.389). The depth limit of 5 prevented overfitting, and the tree can capture non-linear effects and feature interactions, such as the strong effect of recent repayment status (`PAY_0`). Recall (0.35) is still modest. |
| KNN | Mid-range on every metric (accuracy 0.806, F1 0.43, MCC 0.35). With 23 features and an imbalanced target, the neighbours of a point are mostly non-defaulters, which limits recall (0.33). It is also slower at prediction time than the other models. |
| Naive Bayes | Highest recall (0.55) and F1 (0.50), but the lowest accuracy (0.752) and precision (0.45). The independence assumption does not hold here, since the six bill amounts are strongly correlated, so the model over-predicts defaults. It is attractive when missing a defaulter is more costly than a false alarm. |
| Support Vector Machine | Accuracy (0.816) and MCC (0.381) close to the Decision Tree, with good precision (0.67) but low recall (0.34). AUC (0.719) is lower than the tree's. It is the slowest model to train on 24,000 rows. |
| **Overall winner for this dataset** | **Decision Tree.** It has the best Accuracy, AUC and MCC. MCC is the most balanced single metric on imbalanced data, and the tree is also fast and interpretable. If the business goal is to catch as many defaulters as possible, Naive Bayes is the better choice because of its recall (0.55), at the cost of more false alarms. |

**General remark:** accuracy is misleading on this dataset, because always predicting "no default" already gives about 78%. All models have recall below 0.56 because of the 78/22 class imbalance. Techniques such as `class_weight='balanced'`, SMOTE or threshold tuning could improve recall and are left as future work.

## Streamlit app

**Live link:** https://credit-card-default-ml-classifier.streamlit.app/

The app opens with results already visible: when no file is uploaded, it scores the bundled `test_data.csv` (the same 6,000-row test split used for the table above). Uploading a CSV in the sidebar replaces the demo data.

**Required features**

| Requirement | Where in the app |
|---|---|
| a. Dataset upload option (CSV, test data only) | Sidebar: "Upload a test CSV" |
| b. Model selection dropdown | Sidebar: "Classifier" (all 5 models) |
| c. Display of evaluation metrics | Overview page (all models) and Single model page (Accuracy, AUC, Precision, Recall, F1, MCC) |
| d. Confusion matrix and classification report | Single model page |

The uploaded CSV must contain the 23 feature columns and the `default` target column. The app loads the saved models from `model/` and only predicts. **No retraining happens on the uploaded data.**

**Pages**
- **Overview:** best model for Accuracy, AUC, Recall and MCC, the scoreboard of all five models (best value per metric highlighted) and a chart of all metrics side by side.
- **Single model:** the six metrics, the confusion matrix (counts or recall per class) and the classification report for the selected model.
- **ROC curves:** ROC curves of all five models on one plot, with the selected model emphasised.
- **Customer scores:** predicted label and risk score for each customer, riskiest first, with a filter for misclassified rows and a CSV download.
- **What drives defaults:** top 10 most important features for the Decision Tree and Logistic Regression.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py UCI_Credit_Card.csv   # optional: retrain and regenerate the models
streamlit run app.py
```

## Execution environment
<img width="1917" height="960" alt="image (2)" src="https://github.com/user-attachments/assets/a5b152c3-780d-460e-b0ab-bb0c62e8c6b9" />

The assignment was executed on the BITS Virtual Lab.

## App screenshots

**1. Streamlit app output**
<img width="1890" height="877" alt="image" src="https://github.com/user-attachments/assets/41cb76a9-8182-4d06-8232-5ad726beb13a" />

**2. ROC Curves**
<img width="1871" height="755" alt="image" src="https://github.com/user-attachments/assets/6b5eda50-c35f-4b15-9ace-47e62b0c4840" />

**3. Leaderboard**
<img width="1837" height="771" alt="image" src="https://github.com/user-attachments/assets/b780aceb-3a7c-4bcf-a464-8414479ba5d7" />


ML classification of credit card default (UCI dataset) with 5 models and a Streamlit app
