from main import HyperEnsamble
import pandas as pd
import numpy as np

def test_titanic():
    TRAIN = 600
    # load data
    train = pd.read_csv("titanic.csv")
    target = train["Survived"]
    train = train.drop(["PassengerId", "Survived"], axis=1)
    train["Age_nan"] = train["Age"].isnull().apply(lambda x:1 if x else 0)
    train = train.fillna(train.mean())
    train = pd.get_dummies(train, dummy_na=True)

    # define 
    from sklearn.metrics import accuracy_score
    def accuracy_error_func(y_true, y_pred):
        y_pred = np.array([1 if y > 0.5 else 0 for y in y_pred])
        return -accuracy_score(y_true, y_pred)
        
    he = HyperEnsamble(train[:TRAIN],target[:TRAIN], accuracy_error_func,None, trial=10)
    he.find_best_models()
    assert he.find_best_ensamble() < -0.75
    
