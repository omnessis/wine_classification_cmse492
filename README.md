# Wine Classification — CMSE 492

A clean, reproducible machine-learning project that trains and evaluates a
**Logistic Regression** classifier on the
[UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
(also available on [Kaggle](https://www.kaggle.com/datasets/brynja/wineuci)).

This repo now uses a notebook-only workflow: the main end-to-end run lives in
`wine_classification_notebook.ipynb`.

The dataset contains 178 wine samples from three Italian cultivars, each
described by 13 chemical measurements. The task is to predict which cultivar a
wine belongs to.

---

## Project structure

```
wine_classification_cmse492/
├── data/
│   └── wine.csv                  # UCI Wine dataset (178 samples, 13 features)
├── wine_classification_notebook.ipynb  # Main notebook workflow
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

### 4. Run the notebook workflow

Open `wine_classification_notebook.ipynb` and run the cells from top to bottom.
The notebook will:

1. Install the needed packages if necessary
2. Load `data/wine.csv`
3. Split and standardize the data
4. Train a logistic regression model
5. Print accuracy and a classification report
6. Validate the results and save a confusion matrix image

### 5. Optional: inspect the notebook output

After running the notebook, you should see the printed accuracy, the
classification report, and the saved confusion matrix image.

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

1. **Load** `data/wine.csv` in the notebook.
2. **Split** into 80 % training / 20 % test sets (stratified, `random_state=42`).
3. **Standardise** features with `StandardScaler` (fitted on training data only).
4. **Train** `sklearn.linear_model.LogisticRegression` (`max_iter=1000`).
5. **Evaluate** with accuracy score, classification report, and confusion matrix.
6. **Validate** with simple assertions on shape, labels, split size, and accuracy.

---

## Results

| Metric   | Value |
|----------|-------|
| Accuracy | 97.2 % |

The notebook also saves the confusion matrix to `wine_confusion_matrix.png`.

---

## Requirements

- Python ≥ 3.9
- scikit-learn ≥ 1.3.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0
