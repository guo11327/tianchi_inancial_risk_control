#%%
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from preprocess import data_preprocess,add_target_features
warnings.filterwarnings('ignore')
# %%
data_train, target, data_test  = data_preprocess()
#%%

# target encoder
target_encode_cols = ['postCode', 'regionCode', 'homeOwnership', 'employmentTitle','title']
kflod_num = 5
data_train, data_test = add_target_features(data_train,data_test,target,target_encode_cols,kflod_num)
#%%
data_train.to_csv("./data_train.csv",index=None)
data_test.to_csv("./data_test.csv",index=None)

#%%
columns = data_train.columns.tolist()
to_remove = ["id","isDefault"]
for x in to_remove:
    columns.remove(x)

#%%
shold = 1000
saveFeature_df = pd.read_csv('./feature_importance_df_0.csv')
saveFeature_df.columns=['feature','score']
columns = list(saveFeature_df[saveFeature_df['score']>shold]['feature'])

#%%
features = columns
x_train = data_train[features]
x_test = data_test[features]
y_train = target

#%%
best_model = None
best_pred = None
feature_importance_df = pd.DataFrame()
res_list = []

def cv_model(clf, train_x, train_y, test_x, clf_name):
    global best_model
    global best_pred
    global res_list
    global feature_importance_df

    max_score = 0
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    model = None
    test_pred = None
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            # params = {
            #     'boosting_type': 'gbdt',
            #     'objective': 'binary',
            #     'metric': 'auc',
            #     'min_child_weight': 5,
            #     'num_leaves': 2 ** 5,
            #     'lambda_l2': 10,
            #     'feature_fraction': 0.8,
            #     'bagging_fraction': 0.8,
            #     'bagging_freq': 4,
            #     'learning_rate': 0.1,
            #     'seed': 2020,
            #     'nthread': 28,
            #     'n_jobs':24,
            #     'silent': True,
            #     'verbose': -1,
            # }
            params = {
                'boosting_type':'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.01,
                'num_leaves': 14,
                'max_depth': 19,
                'min_data_in_leaf': 37,
                'min_child_weight':1.6,
                'bagging_fraction': 0.98,
                'feature_fraction': 0.69,
                'bagging_freq': 96,
                'reg_lambda': 9,
                'reg_alpha': 7,
                'min_split_gain': 0.4,
                'nthread': 8,
                'seed': 2020,
                'silent': True,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            
            # params = {'booster': 'gbtree',
            #           'objective': 'binary:logistic',
            #           'eval_metric': 'auc',
            #           'gamma': 1,
            #           'min_child_weight': 1,
            #           'max_depth': 6,
            #           'max_delta_step': 1,
            #           'lambda': 10,
            #           'subsample': 0.7,
            #           'colsample_bytree': 1,
            #           'colsample_bylevel': 1,
            #           'eta': 0.3,
            #           'tree_method': 'exact',
            #           'seed': 2020,
            #           "silent": True,
            #           }
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 5,
                      'max_depth': 7,
                    #   'lambda': 10,
                      'subsample': 0.8,
                      'colsample_bytree': 0.4,
                    #   'colsample_bylevel': 1,
                      'eta': 0.01,
                    #   'tree_method': 'exact',
                      'seed': 2021,
                      'nthread': -1,
                      'tree_method': 'gpu_hist',
                    #   'scale_pos_weight':4,
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=200, early_stopping_rounds=600)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_matrix = clf.DMatrix(test_x)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
                 
        if clf_name == "cat":
            params = {
                'learning_rate': 0.05, 
                'depth': 5, 
                'l2_leaf_reg': 10, 
                'bootstrap_type': 
                'Bernoulli',
                'od_type': 'Iter', 
                'od_wait': 50, 
                'random_seed': 11, 
                'allow_writing_files': False}
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test = test + test_pred / kf.n_splits
        cur_score = roc_auc_score(val_y, val_pred)
        res_list.append([cur_score,test_pred])
        cur_importance_df = pd.DataFrame()
        cur_importance_df["Feature"] = model.get_fscore().keys()
        cur_importance_df["importance"] = model.get_fscore().values()
        cur_importance_df["fold"] = i + 1

        feature_importance_df = pd.concat([feature_importance_df, cur_importance_df], axis=0)
        if cur_score > max_score:
            best_model = model
            best_pred = test_pred
            max_score = cur_score
        cv_scores.append(cur_score)
        print(cv_scores)
    
    feature_sorted = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(ascending=False)
    # feature_sorted.to_csv('./feature_importance_df.csv')
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
    return cat_train, cat_test

#%%

lgb_train, lgb_test = xgb_model(x_train, y_train, x_test)

# %%
res = pd.DataFrame(lgb_test,columns=["isDefault"])
name = 0.7412
res["id"]=range(800000,1000000,1)
res.set_index("id",inplace=True)
res.to_csv(f"./data/result/res_{name}.csv")

# %%
def save_result(data,name):
    res = pd.DataFrame(data,columns=["isDefault"])
    res["id"]=range(800000,1000000,1)
    res.set_index("id",inplace=True)
    res.to_csv(f"./data/result/res_{name}.csv")

#%%
series_list = []
for stype in ['weight','gain','cover','total_gain','total_cover']:
    se = pd.Series(best_model.get_score(importance_type = stype)).rename(stype)
    series_list.append(se)
feature_info = pd.concat(series_list,axis=1)

#%%
feature_info.to_csv("./feature_info.csv")
# %%
