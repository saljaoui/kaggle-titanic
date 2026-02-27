import pandas as pd

train = pd.read_csv("../data/train.csv")

def preprocess_data(train, test):
    train = train.copy()
    test = test.copy()

    test_ids = test["PassengerId"].copy()

    median_age = train["Age"].median()
    train["Age"] = train["Age"].fillna(median_age)
    test["Age"] = test["Age"].fillna(median_age)

    test["Fare"] = test["Fare"].fillna(train["Fare"].median())

    mode_emb = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(mode_emb)
    test["Embarked"] = test["Embarked"].fillna(mode_emb)

    train.drop("Cabin", axis=1, inplace=True, errors="ignore")
    test.drop("Cabin", axis=1, inplace=True, errors="ignore")

    train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"
    ]

    train["Title"] = train["Title"].replace(rare_titles, "Rare")
    test["Title"] = test["Title"].replace(rare_titles, "Rare")

    train["Title"] = train["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    test["Title"] = test["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    title_map = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Rare": 4
    }

    train["Title"] = train["Title"].map(title_map)
    test["Title"] = test["Title"].map(title_map)


    # train.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    # test.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    emb_map = {"S": 0, "C": 1, "Q": 2}
    train["Embarked"] = train["Embarked"].map(emb_map)
    test["Embarked"] = test["Embarked"].map(emb_map)

    print(train.head())

    y = train["Survived"]
    X = train.drop("Survived", axis=1)

    return X, y, test, test_ids

X, y, test, test_ids = preprocess_data(train, pd.read_csv("../data/test.csv"))
print(X.head())
print(y.head())
print(test.head())