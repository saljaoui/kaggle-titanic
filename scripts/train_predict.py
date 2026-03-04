#!/usr/bin/env python3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "models" / "titanic_model.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submissions" / "submission.csv"

def _prepare_titles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Dr", "Rev", "Major", "Col", "Don", "Sir", "Capt", "Jonkheer"], "Mr"
    )
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme", "Lady", "Countess", "Dona"], "Mrs")
    return df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    passenger_id = test_df["PassengerId"].copy()

    for df in (train_df, test_df):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["HasCabin"] = df["Cabin"].notna().astype(int)

    train_df = _prepare_titles(train_df)
    test_df = _prepare_titles(test_df)

    title_age_median = train_df.groupby("Title")["Age"].median()
    global_age_median = train_df["Age"].median()
    train_df["Age"] = train_df["Age"].fillna(train_df["Title"].map(title_age_median))
    test_df["Age"] = test_df["Age"].fillna(test_df["Title"].map(title_age_median))
    train_df["Age"] = train_df["Age"].fillna(global_age_median)
    test_df["Age"] = test_df["Age"].fillna(global_age_median)

    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

    embarked_mode = train_df["Embarked"].mode()[0]
    train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
    test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

    for df in (train_df, test_df):
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df["Title"] = (
            df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}).fillna(0).astype(int)
        )

        df["Fare_Per_Person"] = df["Fare"] / df["FamilySize"]
        df["FareLog"] = np.log1p(df["Fare"])
        df["Fare_Per_Person_Log"] = np.log1p(df["Fare_Per_Person"])
        df["Age_Class"] = df["Age"] * df["Pclass"]
        df["Sex_x_Pclass"] = df["Sex"] * df["Pclass"]
        df["Age_Sex"] = df["Age"] * df["Sex"]
        df["Sex_Title"] = df["Sex"] * df["Title"]
        df["IsWoman_1st"] = ((df["Sex"] == 1) & (df["Pclass"] == 1)).astype(int)
        df["IsWoman_3st"] = ((df["Sex"] == 1) & (df["Pclass"] == 3)).astype(int)
        df["IsWoman"] = (df["Sex"] == 1).astype(int)
        df["IsBoy"] = ((df["Age"] < 15) & (df["Sex"] == 0)).astype(int)
        df["IsBoyAlone"] = ((df["Age"] < 15) & (df["FamilySize"] == 1)).astype(int)

        df.drop(
            [
                "Name",
                "Ticket",
                "Cabin",
                "Parch",
                "SibSp",
                "PassengerId",
                "Fare",
                "Fare_Per_Person",
                "FamilySize",
            ],
            axis=1,
            inplace=True,
        )

    y_train = train_df["Survived"]
    x_train = train_df.drop("Survived", axis=1)
    x_test = test_df
    return x_train, y_train, x_test, passenger_id


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    x_train, y_train, x_test, passenger_id = preprocess_data(train_df, test_df)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")
    print(f"Mean CV accuracy: {scores.mean():.6f}")

    model.fit(x_train, y_train)

    OUTPUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("submissions").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    preds = model.predict(x_test)
    submission = pd.DataFrame({"PassengerId": passenger_id, "Survived": preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
