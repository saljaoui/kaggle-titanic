# train_predict.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def preprocess_data(train, test):
    train = train.copy()
    test = test.copy()

    test_ids = test["PassengerId"].copy()

    train["Title"] = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    test["Title"]  = test["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    train["Title"] = train["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    test["Title"]  = test["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    global_median_age = train["Age"].median()
    title_age_median = train.groupby("Title")["Age"].median()

    train["Age"] = train["Age"].fillna(train["Title"].map(title_age_median)).fillna(global_median_age)
    test["Age"]  = test["Age"].fillna(test["Title"].map(title_age_median)).fillna(global_median_age)


    test["Fare"] = test["Fare"].fillna(train["Fare"].median())

    mode_emb = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(mode_emb)
    test["Embarked"] = test["Embarked"].fillna(mode_emb)

    train.drop("Cabin", axis=1, inplace=True, errors="ignore")
    test.drop("Cabin", axis=1, inplace=True, errors="ignore")

    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}
    train["Title"] = train["Title"].map(title_map).fillna(4).astype(int)
    test["Title"] = test["Title"].map(title_map).fillna(4).astype(int)

    train.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    test.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    emb_map = {"S": 0, "C": 1, "Q": 2}
    train["Embarked"] = train["Embarked"].map(emb_map)
    test["Embarked"] = test["Embarked"].map(emb_map)

    y = train["Survived"]
    X = train.drop("Survived", axis=1)

    return X, y, test, test_ids

def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X, y, test_enc, test_ids = preprocess_data(train, test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
        )

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print("CV accuracy:", scores.mean())

    model.fit(X, y)

    joblib.dump(model, "output/models/titanic_model.pkl")
    print("Saved: titanic_model.pkl")

    preds = model.predict(test_enc)

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": preds
    })

    submission.to_csv("output/submissions/submission.csv", index=False)
    print("Saved: submission.csv")

if __name__ == "__main__":
    main()