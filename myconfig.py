import os
root_path = os.path.curdir
root_path = "./"
model_name = "xgboost"
model_path = "model/xgboost/"


data_path = "data/"
origin_data_path = data_path+"origin_data/"
dataset_path = data_path+"dataset/"
data_train_path = dataset_path+"data_train.csv"
data_test_path = dataset_path+"data_test.csv"

result_path = data_path+"result/"
feature_score_path = model_path+"feature_score.csv"

for path in [data_path,"model/",model_path,origin_data_path,dataset_path,result_path]:
    if not os.path.exists(path):
        os.mkdir(path)