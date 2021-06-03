#%%
import os
import pandas as pd
import numpy as np
from myconfig import data_train_path,data_test_path,result_path,feature_score_path
from preprocess import load_dataset
from model.xgbost import xgb_model
#%%
# 
shold = 0.1

# load dataset
def load_data():
    data_train = None
    data_test = None
    if os.path.exists(data_train_path) and os.path.exists(data_test_path):
        data_train = pd.read_csv(data_train_path)
        data_test = pd.read_csv(data_test_path)
        print(f"load dataset from file.")
    else:
        print(f"dataset file does not exist, begin generate....")
        data_train, data_test = load_dataset()
        print(f"dataset generate down")
    return data_train, data_test

def transform_data(data_train,data_test):
    target  = data_train["isDefault"]
    columns = data_train.columns.tolist()
    to_remove = ["id","isDefault"]
    # remove unimport cols base feature info.
    if os.path.exists(feature_score_path):
        df = pd.read_csv(feature_score_path,index_col=0)
        features = df.groupby(by=['Feature'],as_index=True)['importance'].mean().sort_values()
        to_remove.extend(features.head(int(features.shape[0]*shold)).index.tolist())

    for x in to_remove:
        if x in columns:
            columns.remove(x)

    features = list(columns)
    x_train = data_train[features]
    x_test = data_test[features]
    y_train = target
    print(f"data shape : {x_train.shape}")
    return x_train,x_test,y_train

#%%
if __name__ == "__main__":
    data_train, data_test = load_data()
    
    x_train,x_test,y_train = transform_data(data_train, data_test)

    lgb_train, lgb_test = xgb_model(x_train, y_train, x_test)

    res = pd.DataFrame(lgb_test,columns=["isDefault"])
    name = "lgb"
    
    res["id"]=range(800000,1000000,1)
    res.set_index("id",inplace=True)
    res.to_csv(result_path+f"res_{name}.csv")
    

    
# %%
