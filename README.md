# Titanic Survival Prediction (Kaggle)

## Introduction
This project solves the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition task: predict whether each passenger survived (`Survived = 1`) or not (`Survived = 0`) using passenger information.

Project goal: build a strong baseline and target at least **78.9%** Kaggle accuracy.

## Username
`Saljaoui`

## Project Structure
```text
kaggle-titanic/
├─ README.md
├─ requirements.txt
├─ username.txt
├─ .gitignore
├─ data/
│  ├─ train.csv
│  ├─ test.csv
│  └─ gender_submission.csv
├─ notebooks/
│  ├─ EDA.ipynb
│  └─ main.ipynb
└─ output/
   ├─ report/
   │  └─ baseline_notes.md
   └─ submissions/
      └─ submission.csv
```

## Data Preprocessing
Implemented in `notebooks/main.ipynb` (`preprocess_data`):

1. Missing values:
   - `Age`: filled in train and test with the **train median age**.
   - `Fare` (test only): filled with the **train median fare**.
   - `Embarked`: filled in train and test with the **train mode**.
2. Dropped columns:
   - `Cabin` (high missingness).
   - `PassengerId`, `Name`, `Ticket` (not used as model features).
3. Target/feature split:
   - Target: `Survived`
   - Features: remaining processed columns.

## Feature Engineering
The project applies lightweight, explicit feature engineering:

1. Categorical encoding:
   - `Sex`: `male -> 0`, `female -> 1`
   - `Embarked`: `S -> 0`, `C -> 1`, `Q -> 2`
2. Final feature set:
   - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

No additional derived features (for example, title extraction, family-size features, or ticket group features) are implemented in the current baseline.

## Model And Validation Strategy
Implemented in `notebooks/main.ipynb`:

- Model: `LogisticRegression(max_iter=1000)`
- Validation: `cross_val_score(..., cv=5, scoring="accuracy")`
- Training flow:
  1. preprocess data
  2. evaluate with 5-fold cross-validation
  3. fit model on full training data
  4. predict on test data
  5. write submission file

## Scores
- Cross-validation accuracy (mean, 5-fold): **0.7934906785512524** (~79.35%)
- Kaggle public leaderboard score: **0.78xx** is mentioned in `output/report/baseline_notes.md` (exact score not recorded)
- Exact Kaggle score placeholder: **`0.78xx`**

## How To Run
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter:
   ```bash
   jupyter notebook
   ```
4. Run `notebooks/main.ipynb` from top to bottom.
5. Get predictions at:
   - `output/submissions/submission.csv`
6. Upload `submission.csv` to Kaggle for leaderboard evaluation.

## Requirements
From `requirements.txt`:

- `numpy`: numerical operations
- `pandas`: data loading and tabular preprocessing
- `matplotlib`: simple visualization of prediction distribution
- `scikit-learn`: logistic regression model and cross-validation

## Conclusion
This repository implements a clean baseline Titanic pipeline using straightforward preprocessing, label encoding, and Logistic Regression with 5-fold cross-validation. The baseline reaches about **79.35% CV accuracy** and around the **high-0.78 Kaggle range**, establishing a solid starting point for further feature engineering and model improvements.
