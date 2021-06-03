#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from myconfig import feature_score_path
warnings.filterwarnings('ignore')


def cv_model(clf, train_x, train_y, test_x, clf_name):
    feature_importance_df = pd.DataFrame()
    max_score = 0
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    model = None
    test_pred = None
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 2,
                      'min_child_weight': 5,
                      'max_depth': 5,
                      'subsample': 0.8,
                    #   'colsample_bytree': 0.4,
                    # #   'colsample_bylevel': 1,
                      'eta': 0.01,
                      'alpha': 1,
                      'lambda': 1,
                    # #   'tree_method': 'exact',
                      'seed': 2021,
                      'nthread': -1,
                      'tree_method': 'gpu_hist',
                    #   'scale_pos_weight':4,  无明显提升
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=200, early_stopping_rounds=600)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_matrix = clf.DMatrix(test_x)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
            
        train[valid_index] = val_pred
        test = test + test_pred / kf.n_splits
        cur_score = roc_auc_score(val_y, val_pred)
        cur_importance_df = pd.DataFrame()
        cur_importance_df["Feature"] = model.get_fscore().keys()
        cur_importance_df["importance"] = model.get_fscore().values()
        cur_importance_df["fold"] = i + 1

        feature_importance_df = pd.concat([feature_importance_df, cur_importance_df], axis=0)
        cv_scores.append(cur_score)
        print(cv_scores)
    
    # feature_sorted = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(ascending=False)
    # feature_sorted.to_csv('./feature_importance_df.csv')
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    feature_importance_df.to_csv(feature_score_path)
    return train, test


def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test

# %%
# lgb_train, lgb_test = xgb_model(x_train, y_train, x_test)
# # %%
# res = pd.DataFrame(lgb_test,columns=["isDefault"])
# name = "0.7411"

# res["id"]=range(800000,1000000,1)
# res.set_index("id",inplace=True)
# res.to_csv(result_path+f"res_{name}.csv")
# %%
