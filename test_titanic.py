from main import HyperEnsamble
import pandas as pd
import numpy as np
import main
import glob
import shutil

def test_titanic(test_dir="test_history"):
    shutil.rmtree(test_dir)
    TRAIN = 600
    # load data
    train = pd.read_csv("titanic.csv")
    target = train["Survived"]
    train = train.drop(["PassengerId", "Survived", "Name"], axis=1)
    train["Age_nan"] = train["Age"].isnull().apply(lambda x:1 if x else 0)
    train = train.fillna(train.mean())
    train = pd.get_dummies(train, dummy_na=True)

    # define loss function
    from sklearn.metrics import accuracy_score
    def accuracy_error_func(y_true, y_pred):
        y_pred = np.array([1 if y > 0.5 else 0 for y in y_pred])
        return -accuracy_score(y_true, y_pred)
    
    num_trial = 5
    he = HyperEnsamble(train[:TRAIN],target[:TRAIN], accuracy_error_func,None, test_dir, trial=num_trial)
    he.find_best_models()
    assert he.find_best_ensamble() < -0.76
    validation_error = accuracy_error_func(target[TRAIN:],he.predict(train[TRAIN:]))
    print("final validation:",validation_error)
    assert validation_error < -0.76
    
    # testing saved models
    expected_saved_model = num_trial * len(he.model_confs)
    num_cached_model = len(he.mh.get_all_model()) 

    # reusing the result
    he = HyperEnsamble(train[:TRAIN], target[:TRAIN], accuracy_error_func, None, test_dir, trial=1, model_confs = [main.SvmConf])
    he.find_best_models()
    assert len(he.mh.get_all_model()) == num_cached_model 
    assert len(list(glob.glob(he.mh.path_feature + "/*"))) == 1

    # need to train because feature changed
    train = train.drop(["Fare"], axis=1)
    he = HyperEnsamble(train[:TRAIN], target[:TRAIN], accuracy_error_func, None, test_dir, trial=1, model_confs = [main.SvmConf])
    he.find_best_models()
    assert len(he.mh.get_all_model()) == num_cached_model + 1
    assert len(list(glob.glob(he.mh.path_feature + "/*"))) == 2

if __name__=="__main__":
    test_titanic()