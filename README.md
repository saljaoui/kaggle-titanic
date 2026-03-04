# Titanic Survival Prediction (Kaggle)

## Introduction
This project solves the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) task: predict whether each passenger survived (`Survived = 1`) or not (`Survived = 0`).

Current training and submission pipeline is implemented in:
- `scripts/train_predict.py`

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
├─ output/
│  ├─ models/
│  │  └─ titanic_model.pkl
│  ├─ report/
│  │  └─ baseline_notes.md
│  └─ submissions/
│     └─ submission.csv
└─ scripts/
   └─ train_predict.py
```

## Data Preprocessing
Implemented in `scripts/train_predict.py` (`preprocess_data`):

1. `Title` extraction from `Name` using regex.
2. Title normalization:
   - `Mlle -> Miss`
   - `Ms -> Miss`
   - `Mme -> Mrs`
3. Missing values:
   - `Age`: filled with title-based median age from train, then global train median as fallback.
   - `Fare` (test only): filled with train median fare.
   - `Embarked`: filled with train mode.
4. Dropped columns:
   - `Cabin`
   - `PassengerId`, `Name`, `Ticket`
5. Encoding:
   - `Sex`: `male -> 0`, `female -> 1`
   - `Embarked`: `S -> 0`, `C -> 1`, `Q -> 2`
   - `Title`: `Mr -> 0`, `Miss -> 1`, `Mrs -> 2`, `Master -> 3`, others -> `4`
6. Target/feature split:
   - Target: `Survived`
   - Features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Title`

## Model And Validation Strategy
Implemented in `scripts/train_predict.py`:

- Model: `GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=3, max_features="sqrt", random_state=RANDOM_STATE)`
- Validation: `cross_val_score(..., cv=5, scoring="accuracy")`
- Training flow:
  1. load `data/train.csv` and `data/test.csv`
  2. preprocess train/test data
  3. evaluate with 5-fold cross-validation
  4. fit model on full training data
  5. save model to `output/models/titanic_model.pkl`
  6. predict on test data
  7. save submission to `output/submissions/submission.csv`

## Scores
Latest local run output:
- Cross-validation accuracy (mean, 5-fold): **0.8249074132195091**

Kaggle leaderboard score is not documented in this repository yet.

## How To Run
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training + prediction:
   ```bash
   python3 scripts/train_predict.py
   ```
4. Generated files:
   - Model: `output/models/titanic_model.pkl`
   - Submission: `output/submissions/submission.csv`

## Requirements
From `requirements.txt`:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Conclusion
The current baseline uses feature engineering with passenger title extraction and a RandomForest model. This pipeline reaches **0.8249 CV accuracy** and generates a ready-to-upload Kaggle submission file.
