# Baseline Model Notes

## Objective
Build a stronger baseline model for the Kaggle Titanic competition using feature engineering and Random Forest.

---

## Preprocessing

The following steps were applied:

- Extracted **Title** from passenger **Name**.
- Normalized titles:
  - Mlle -> Miss
  - Ms -> Miss
  - Mme -> Mrs
- Filled missing **Age** using title-based median age from train, then fallback to global train median.
- Filled missing **Fare** values in test using train median.
- Filled missing **Embarked** values using train mode.
- Dropped **Cabin**.
- Dropped **Name**, **Ticket**, and **PassengerId** after keeping PassengerId for final submission.
- Encoded:
  - Sex → {male: 0, female: 1}
  - Embarked → {S: 0, C: 1, Q: 2}
  - Title → {Mr: 0, Miss: 1, Mrs: 2, Master: 3, Other: 4}

Final feature set:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Title

---

## Model

Model used:
- Random Forest
- n_estimators = 200
- max_depth = 5
- random_state = 42
- 5-fold cross-validation

---

## Performance

Cross-validation accuracy:
> 0.8249074132195091

Kaggle public leaderboard score:
> Not recorded yet in this report.

---

## Observations

- Title-based age imputation improves handling of missing `Age`.
- Adding `Title` as a feature provides additional social-status signal.
- This baseline reaches about 82.49% CV accuracy locally.
