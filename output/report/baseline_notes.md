# Baseline Model Notes

## Objective
Build a simple baseline model for the Kaggle Titanic competition using basic preprocessing and Logistic Regression.

---

## Preprocessing

The following steps were applied:

- Filled missing **Age** values using the median.
- Filled missing **Fare** values in the test set using the train median.
- Filled missing **Embarked** values using the mode.
- Dropped **Cabin** (too many missing values).
- Dropped **Name**, **Ticket**, and **PassengerId** (not used as features).
- Encoded:
  - Sex → {male: 0, female: 1}
  - Embarked → {S: 0, C: 1, Q: 2}

Final feature set:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## Model

Model used:
- Logistic Regression
- max_iter = 1000
- 5-fold cross-validation

---

## Performance

Cross-validation accuracy:
> 0.7935

Kaggle public leaderboard score:
> 0.78xx

---

## Observations

- Sex is a strong predictor (female survival rate is much higher).
- First-class passengers have higher survival rates.
- Even a simple baseline model achieves close to 80% accuracy.