# Wine Classification — CMSE 492

A clean, reproducible machine-learning project that trains and evaluates a
**Logistic Regression** classifier on the
[UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
(also available on [Kaggle](https://www.kaggle.com/datasets/brynja/wineuci)).

The dataset contains 178 wine samples from three Italian cultivars, each
described by 13 chemical measurements. The task is to predict which cultivar a
wine belongs to.

---

## Project structure

```
wine_classification_cmse492/
├── data/
│   └── wine.csv                  # UCI Wine dataset (178 samples, 13 features)
├── src/
│   └── wine_classification.py    # Load → preprocess → train → evaluate
├── tests/
│   └── test_wine_classification.py  # 13 pytest unit tests
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/omnessis/wine_classification_cmse492.git
cd wine_classification_cmse492
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the classifier

```bash
python src/wine_classification.py
```

Expected output:

```
Dataset loaded: 178 samples, 13 features
Train samples: 142 | Test samples: 36
Model training complete.

Test Accuracy: 0.9722

Classification Report:
              precision    recall  f1-score   support

  Cultivar 1       1.00      1.00      1.00        12
  Cultivar 2       0.93      1.00      0.97        14
  Cultivar 3       1.00      0.90      0.95        10

    accuracy                           0.97        36
   ...

Confusion matrix saved to confusion_matrix.png
```

### 5. Run the tests

```bash
python -m pytest tests/ -v
```

---

## Dataset

| Property        | Value |
|-----------------|-------|
| Samples         | 178   |
| Features        | 13    |
| Classes         | 3     |
| Missing values  | None  |

**Features:** Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium,
Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins,
Color intensity, Hue, OD280/OD315 of diluted wines, Proline.

**Target:** Cultivar identifier (1, 2, or 3).

> All features are continuous. Standardisation is applied before training,
> as recommended for scale-sensitive classifiers.

---

## Methodology

1. **Load** `data/wine.csv`.
2. **Split** into 80 % training / 20 % test sets (stratified, `random_state=42`).
3. **Standardise** features with `StandardScaler` (fitted on training data only).
4. **Train** `sklearn.linear_model.LogisticRegression` (`max_iter=1000`).
5. **Evaluate** with accuracy score, classification report, and confusion matrix.

---

## Results

| Metric   | Value |
|----------|-------|
| Accuracy | 97.2 % |

---

## Requirements

- Python ≥ 3.9
- scikit-learn ≥ 1.3.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0
