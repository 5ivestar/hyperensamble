import pandas as pd
import numpy as np
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
from hyperopt.pyll import scope
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import sklearn.utils
import os
import glob
import hashlib

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

from sklearn.svm import SVC
class SvmConf(ModelConf):
    param_space = {
        'C': hp.uniform('C', 0.1, 2.0),
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': scope.int(hp.quniform('degree', 2, 5, 1)),
        'gamma': hp.choice('gamma', ['auto', 'scale']),
        'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
        'max_iter': scope.int(hp.quniform('max_iter', -1, 100, 1))
    }
    name = "svm_classifier"
    def instance(self, param):
        return SVC(**param)

from sklearn.naive_bayes import GaussianNB
class NaiveBayseConf(ModelConf):
    param_space = {
    }
    name = "naive_bayse"
    def instance(self, param):
        return GaussianNB(**param)

class LinearRegressionConf(ModelConf):
    param_space = {}

    def instance(self, param):
        return LinearRegression(normalize=True)

class HyperEnsamble:
    def __init__(self, df, target, error_func, pred_type, history_root,trial=10, model_confs=None):
        npdf = df.values if isinstance(df, pd.DataFrame) else df
        train = df[:len(target)]
        self.train = npdf[:len(target)]
        self.test = npdf[len(target):]
        self.target = target 
        if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
            self.target = self.target.values
        if model_confs:
            self.model_confs = model_confs
        else:
            self.model_confs = [XgbConf, LgbmConf, LogisticConf, RandomForestConf, KnnConf, SvmConf, NaiveBayseConf]
        self.error_func = error_func
        self.trial = trial
        self.pred_type = pred_type
        self.mh = ModelHistory(history_root)
        self.train_hash, self.target_hash = self.mh.cacl_pd_hash(train, self.target)
        self.mh.may_store_feature(train, self.train_hash)
        #TODO need to consider shuffle for time
        self.train, self.target = sklearn.utils.shuffle(self.train, self.target, random_state=41)

    def cvtrain(self, params, model_conf, history, x, y, load=True, return_model=False):
        model = model_conf.instance(params)
        kf=KFold(n_splits=2)
        errors=[]
        train_results=[]
        predictions = self.mh.load_past_model_cv(self.train_hash, params, model_conf.name) if load else None
        if predictions is None: 
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
            model.fit(x,y)
            if load:
                test_prediction = model.predict(self.test)
                self.mh.save_model_predict(self.train_hash, params, model_conf.name, predictions, test_prediction)
        error = self.error_func(y, predictions)
        history.append((error, params, predictions))
        return {"loss":error,"status": STATUS_OK} if not return_model else ({"loss":error,"status": STATUS_OK}, model)
        
    def find_best_models(self):
        self.best_model_params = []
        self.best_model_predictions = []
        for model_conf in self.model_confs:
            print("Start training...", model_conf.name)
            # find best params
            history = []
            if model_conf.param_space: #TODO check if param_space contains hyperopt's variable
                def score(params):
                    return self.cvtrain(params, model_conf(self.pred_type), history, self.train, self.target)
                fmin(score, model_conf.param_space, algo=tpe.suggest, trials=Trials(), max_evals=self.trial, rstate=np.random.RandomState(42))
            else:
                self.cvtrain(model_conf.param_space, model_conf(self.pred_type), history, self.train, self.target)
            
            # retrain with best model
            error, best_param, predictions = min(history, key=lambda x:x[0])
            print(model_conf.name, "best error: ", error)
            self.best_model_predictions.append(predictions)
            self.best_model_params.append(best_param)
    
    def find_best_session_ensamble(self):
        history = []
        num_models = len(self.best_model_params)
        best_meta_model = float("inf"), -1
        meta_model_conf = LogisticConf
        for i in range(2**num_models + 1, 2**(num_models+1)):
            selected = np.column_stack([self.best_model_predictions[mid] for mid in range(num_models) if bin(i)[2:][mid] == "1"])
            result = self.cvtrain({}, LogisticConf(self.pred_type), history, selected, self.target, load=False)
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
        print("###session ensamble weight")
        for i,mid in enumerate(sorted(self.selected_model_id)):
            print(self.model_confs[mid].name, self.best_meta_model.coef_[i])
        return best_loss
    
    def find_bigensamble(self, trial):
        model_list = self.mh.get_all_model()
        param_space = {model_path:hp.choice(model_path, [True, False]) for model_path in model_list}
        best_bigensamble = float("inf"), {}, None
        def score(param):
            nonlocal best_bigensamble
            # each model's predictions are feature for stacking model
            features = np.column_stack([self.mh.load_model_cv(model_path) for model_path, use in param.items() if use])
            result, model = self.cvtrain({}, LogisticConf(self.pred_type), [], features, self.target, load=False, return_model=True)
            best_bigensamble = min((result["loss"], param, model), best_bigensamble, key=lambda x:x[0])
            return result
        fmin(score, param_space, algo=tpe.suggest, trials=Trials(), max_evals=trial)
        loss, param, model = best_bigensamble
        print("###best big ensamble loss: ", loss)

        # lastly,  return test_prediction
        test_features = np.column_stack([self.mh.load_model_test_prediction(model_path) for model_path, use in param.items() if use])
        return model.predict(test_features)

    def predict(self, x):
        best_model_prediction = np.column_stack([model.predict(x.values) for i, model in enumerate(self.best_models)])
        return self.best_meta_model.predict(best_model_prediction)

import pickle
import hashlib
class ModelHistory:
    base = 10**9 + 7

    def __init__(self, history_root):
        self.path_root = history_root
        self.path_feature = self.path_root + "/features"
        self.path_model = self.path_root + "/models"
        self.path_exec_record = self.path_root + "/exec_records"
        for path in [self.path_feature, self.path_model, self.path_exec_record]:
            if not os.path.exists(path):
                os.makedirs(path)

    def cacl_pd_hash(self, train, target):
        train_hash = str(myhash(train.columns.tolist()) + len(train))
        target_hash = str(abs(sum(target) + len(target.shape)))
        return train_hash, target_hash

    def may_store_feature(self, train, train_hash):
        if os.path.exists(train_hash):
            return
        train.to_csv(self.path_feature + "/" + train_hash, compression="gzip")
    
    def load_past_model_cv(self, train_hash, params, model_name):
        model_path = self.get_model_path(train_hash, params,model_name)
        if os.path.exists(model_path):
            return np.load(model_path + "/cv_result.npy")
        return None

    def load_model_cv(self, model_path):
        return np.load(model_path + "/cv_result.npy")

    def load_model_test_prediction(self, model_path):
        return np.load(model_path + "/test_prediction.npy")
    
    def save_model_predict(self, train_hash, params, model_name, cv_result, test_prediction):
        model_path = self.get_model_path(train_hash, params, model_name)
        if os.path.exists(model_path):
            return
        os.makedirs(model_path)
        np.save(model_path + "/cv_result", cv_result)
        np.save(model_path + "/test_prediction", test_prediction)

    def get_model_path(self, train_hash, params, model_name):
        param_hash = str(myhash(params))
        model_path = self.path_model + "/" + model_name + "_" + train_hash + "_" + param_hash
        return model_path

    def get_all_model(self):
        return [os.path.abspath(p) for p in glob.glob(self.path_model + "/*")]

def myhash(s):
    if type(s) == str:
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(),16) % 10**8
    elif isinstance(s, list) or isinstance(s, tuple):
        return sum(myhash(e) for e in s) % 10**8
    elif isinstance(s, dict):
        return sum(myhash(k) * 10**4 + 7 + myhash(v) for k,v in s.items()) % 10**8
    else:
        return int(s) % 10**8
