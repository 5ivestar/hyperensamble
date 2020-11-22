import pandas as pd
import numpy as np
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
from hyperopt.pyll import scope
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import sklearn.utils


class ModelConf:
    
    def __init__(self, pred_type):
        self.pred_type = pred_type
    
    def instance(self, param):
        pass

import xgboost as xgb
class XgbConf(ModelConf):
    param_space = {
        'learning_rate':    hp.quniform('learning_rate', 0.05, 0.31, 0.05),
        'max_depth':        scope.int(hp.quniform('max_depth', 5, 16, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.8, 0.1),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'n_estimators':     100,
    }
    name = "xgboost"
    def instance(self, param):
        return xgb.XGBRegressor(**param)

from sklearn.ensemble.forest import RandomForestClassifier
class RandomForestConf(ModelConf):
    param_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'max_features': scope.int(hp.quniform('max_features', 1, 150, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 1)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    }
    name = "random_forest"
    def instance(self, param):
        return RandomForestClassifier(**param)


import lightgbm as lgbm
class LgbmConf(ModelConf):
    param_space = {
        'learning_rate':    hp.quniform('learning_rate',0.05, 0.31, 0.05),
        'max_depth':        scope.int(hp.quniform('max_depth', 5, 16, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.8, 0.1),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'n_estimators':     100,
    }
    name = "lightgbm"
    def instance(self, param):
        return lgbm.LGBMClassifier(**param)

from sklearn.linear_model import LogisticRegression
class LogisticConf(ModelConf):
    param_space ={
        #"penalty": hp.choice("penalty", ["l1","l2","elasticnet","none"]),
        "C": hp.loguniform("C",0.001,100),
        "max_iter":500
    }
    name = "logistic_regression"
    def instance(self, param):
        return LogisticRegression(**param)

from sklearn.neighbors import KNeighborsClassifier
class KnnConf(ModelConf):
    param_space = {
        "n_neighbors": scope.int(hp.quniform("n_neighbors", 1, 50, 1))
    }
    name = "kneibors_classifier"
    def instance(self, param):
        return KNeighborsClassifier(**param)

class LinearRegressionConf(ModelConf):
    param_space = {}

    def instance(self, param):
        return LinearRegression(normalize=True)

class HyperEnsamble:
    def __init__(self, train, target, error_func, pred_type, trial=10, model_confs=None):
        self.train = train.values if isinstance(train, pd.DataFrame) else train
        self.target = target 
        if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
            self.target = self.target.values
        if model_confs:
            self.model_confs = model_confs
        else:
            self.model_confs = [XgbConf, LgbmConf, LogisticConf, RandomForestConf, KnnConf]
        self.error_func = error_func
        self.trial = trial
        self.pred_type = pred_type
        #TODO need to consider shuffle for time
        self.train, self.target = sklearn.utils.shuffle(self.train, self.target, random_state=41)

    def cvtrain(self, params, model_conf, history, x, y):
        model = model_conf.instance(params)
        kf=KFold(n_splits=2)
        errors=[]
        train_results=[]
        predictions = []
        for tr_idx,va_idx in kf.split(x):
            tr_x, va_x = x[tr_idx], x[va_idx]
            tr_y, va_y = y[tr_idx], y[va_idx]
            model.fit(tr_x, tr_y)
            prediction = model.predict(va_x)
            predictions.append(prediction)
            #print(self.error_func(tr_y,model.predict(tr_x)))
            #print(self.error_func(va_y,prediction))
        predictions = np.hstack(predictions)
        score = self.error_func(y, predictions)
        history.append((score, params, predictions))
        return {"loss":score,"status": STATUS_OK}
        
    def find_best_models(self):
        self.best_model_params = []
        self.best_model_predictions = []
        for model_conf in self.model_confs:
            # find best params
            history = []
            def score(params):
                return self.cvtrain(params, model_conf(self.pred_type), history, self.train, self.target)
            fmin(score, model_conf.param_space, algo=tpe.suggest, trials=Trials(), max_evals=self.trial)
            
            # retrain with best model
            score, best_param, predictions = max(history, key=lambda x:x[0])
            self.best_model_predictions.append(predictions)
            self.best_model_params.append(best_param)
    
    def find_best_ensamble(self):
        history = []
        num_models = len(self.best_model_params)
        best_meta_model = float("inf"), -1
        meta_model_conf = LogisticConf
        for i in range(2**num_models + 1, 2**(num_models+1)):
            selected = np.column_stack([self.best_model_predictions[mid] for mid in range(num_models) if bin(i)[2:][mid] == "1"])
            result = self.cvtrain({}, LogisticConf(self.pred_type), history, selected, self.target)
            best_meta_model = min((result["loss"], i), best_meta_model) 
        best_loss, selected_bit = best_meta_model
        print("###best stacking model loss: ", best_loss)

        # retrain with best meta model
        self.selected_model_id = {mid for mid in range(num_models) if bin(selected_bit)[2:][mid] == "1"}
        selected = np.column_stack([self.best_model_predictions[mid] for mid in range(num_models) if mid in self.selected_model_id])
        self.best_meta_model = LinearRegression()
        self.best_meta_model.fit(selected, self.target) 
        self.best_models = []
        for mid in self.selected_model_id:
            best_model = self.model_confs[mid](self.pred_type).instance(self.best_model_params[mid])
            best_model.fit(self.train, self.target)
            self.best_models.append(best_model)
        print("###ensamble weight")
        for i,mid in enumerate(sorted(self.selected_model_id)):
            print(self.model_confs[mid].name, self.best_meta_model.coef_[i])
        return best_loss
    
    def predict(self, x):
        best_model_prediction = np.column_stack([model.predict(x.values) for i, model in enumerate(self.best_models)])
        return self.best_meta_model.predict(best_model_prediction)


