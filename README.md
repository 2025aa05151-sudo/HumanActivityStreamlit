
# Machine Learning Assignment 2

## a) Problem Statement

The objective of this project is to implement multiple classification models 
to predict human activities using sensor-based data and compare their performance.

---

## b) Dataset Description

The dataset used is a multi-class classification dataset containing  561 features and 10299 instances. The goal is to classify human activity 
into six categories.
Since it has a lot of features(561), have trimmd them down and using top 30 only.(using random forests)
Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

## c) Model Comparison Table

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8677 | 0.9842 |      0.8700 |   0.8651 | 0.8659 | 0.8415 |
| Decision Tree       |     0.8263 | 0.9126 |      0.8285 |   0.8229 | 0.8240 | 0.7918 |
| KNN                 |     0.8616 | 0.9651 |      0.8628 |   0.8572 | 0.8584 | 0.8341 |
| Naive Bayes         |     0.7652 | 0.9673 |      0.7848 |   0.7556 | 0.7574 | 0.7225 |
| Random Forest       |     0.8354 | 0.9771 |      0.8396 |   0.8328 | 0.8350 | 0.8025 |
| XGBoost             |     0.8622 | 0.9813 |      0.8635 |   0.8600 | 0.8610 | 0.8347 |

---

## Observations on Model Performance

| ML Model Name       | Observation about model performance                                                                             |
|:--------------------|:----------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Accuracy=0.8677, F1=0.8659, MCC=0.8415. Linear model performs competitively on structured features.             |
| Decision Tree       | Accuracy=0.8263, F1=0.8240, MCC=0.7918. Tree model provides good interpretability with moderate generalization. |
| KNN                 | Accuracy=0.8616, F1=0.8584, MCC=0.8341. Distance-based model depends heavily on feature scaling.                |
| Naive Bayes         | Accuracy=0.7652, F1=0.7574, MCC=0.7225. Probabilistic model assumes feature independence.                       |
| Random Forest       | Accuracy=0.8354, F1=0.8350, MCC=0.8025. Ensemble model shows strong and stable performance.                     |
| XGBoost             | Accuracy=0.8622, F1=0.8610, MCC=0.8347. Ensemble model shows strong and stable performance.                     |

---

All six required models were implemented:
Logistic Regression, Decision Tree, KNN, Naive Bayes,
Random Forest (Ensemble), and XGBoost (Ensemble).

