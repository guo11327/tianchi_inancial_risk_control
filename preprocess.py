#%%
import os
import pandas as pd
from sklearn.preprocessing import Normalizer
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold,KFold
import pickle
from myconfig import model_path,data_path,origin_data_path
from utils import MeanEncoder
from category_encoders import CountEncoder
import time
from tqdm import tqdm
from myconfig import data_train_path,data_test_path
#%%
seed = 2021

count_encode_cols = ['subGrade', 'grade', 'postCode', 'regionCode','homeOwnership','title','employmentTitle','employmentLength']

mean_encode_cols = ['postCode', 'regionCode', 'homeOwnership', 'employmentTitle','title']

target_encode_cols = ['postCode', 'regionCode', 'homeOwnership', 'employmentTitle','title']


month_dict = {'Aug': 8, 'May': 5, 'Jul': 7, 'Oct': 10, 'Dec': 12, 'Apr': 4, 'Jan': 1, 'Nov': 11, 'Feb': 2,
                  'Mar': 3, 'Jun': 6, 'Sep': 9}

pre_model_files = {
    "grade": model_path+"label_encoder_grade.pickle",
    "subGrade" : model_path+"label_encoder_subGrade.pickle",
    "employmentTitle": model_path+"label_encoder_employmentTitle.pickle",
    "postCode": model_path+"label_encoder_postCode.pickle",
    "title": model_path+"label_encoder_title.pickle",
}

def _train_label_encoder(data: pd.Series):
    if data.name not in pre_model_files:
        print(f"{data.name} not support.")
        return
    le = LabelEncoder()
    le.fit(data.astype(str).values)
    with open(pre_model_files[data.name],"wb") as f:
        pickle.dump(le, f)
        print(f"dump {data.name} down.")
    return le

def _load_label_encoder(file):
    le = None
    with open(file,"rb") as f:
        le = pickle.load(f)
        print(f"load {file} down.")
    return le

def _transform_data(data: pd.Series):
    le = _load_label_encoder(pre_model_files[data.name])
    return le.transform(data.astype(str).values)

def deal_loanAmnt(data: pd.Series):
    res = data/1000
    # res = res.astype(int)
    # X = res.values.reshape((1,-1))
    # transformer = Normalizer().fit(X) # fit does nothing.
    # X = transformer.transform(X)
    # return pd.Series(X.squeeze(0))
    return res

def deal_term(data: pd.Series):
    # data.replace({5:1,3:0})
    return data

def deal_interestRate(data: pd.Series):
    res = data.astype(int)
    return res

#%%
def deal_installment(data: pd.Series):
    res = data.astype(int)
    return res//100

def deal_grade(data: pd.Series):
    return _transform_data(data)

def deal_subGrade(data: pd.Series):
    return _transform_data(data)

def deal_employmentTitle(data: pd.Series):
    return _transform_data(data)

def deal_employmentLength(data: pd.Series):
    data.fillna("-1 years",inplace=True)
    data.replace(to_replace="10+ years",value="10 years",inplace=True)
    data.replace(to_replace="< 1 year",value="0 years",inplace=True)

    def deal(x):
        if pd.isnull(x):
            return x
        else:
            return np.int8(x.split(" ")[0])+1

    return data.apply(deal)

def deal_homeOwnership(data: pd.Series):
    return data

def deal_annualIncome(data: pd.Series):
    """cut data 
    """
    bins = [0,10000,30000,80000,150000,300000,500000,100000000]
    labels = range(0,len(bins)-1)
    res = pd.cut(data,bins=bins,labels=labels,right=False).astype(int)
    return res

def deal_verificationStatus(data: pd.Series):
    return data

def deal_issueDate(data : pd.Series):
    data = data.to_frame()
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days
    return data['issueDateDT']

def deal_purpose(data: pd.Series):
    return data

def deal_postCode(data: pd.Series):
    return _transform_data(data)

def deal_regionCode(data: pd.Series):
    return data

def deal_dti(data: pd.Series):
    data = data.astype(int)
    data.replace({-1:0},inplace=True)
    return data // 100

def deal_delinquency_2years(data: pd.Series):
    return data

def deal_ficoRangeLow(data: pd.Series):
    bins =  [0,580,670,740,800,1000]
    labels = range(0,len(bins)-1)
    return pd.cut(data,bins=bins,labels=labels,right=False).astype(int)

def deal_ficoRangeHigh(data: pd.Series):
    bins =  [0,580,670,740,800,1000]
    labels = range(0,len(bins)-1)
    return pd.cut(data,bins=bins,labels=labels,right=False).astype(int)

def deal_openAcc(data: pd.Series):
    return data

def deal_pubRec(data: pd.Series):
    return data

def deal_pubRecBankruptcies(data: pd.Series):
    return data 

def deal_revolBal(data: pd.Series):
    return data

def deal_revolUtil(data: pd.Series):
    return data

def deal_totalAcc(data: pd.Series):
    return data 

def deal_initialListStatus(data: pd.Series):
    return data

def deal_applicationType(data: pd.Series):
    return data

def deal_earliesCreditLine(data : pd.Series):
    res = data.apply(lambda x: np.int(x[-4:]))
    return res

def deal_title(data: pd.Series):
    return _transform_data(data)

def deal_policyCode(data: pd.Series):
    return data



func_map = {
    "loanAmnt": deal_loanAmnt,
    "term" : deal_term,
    "interestRate": deal_interestRate,
    "installment": deal_installment,
    "grade": deal_grade,
    "subGrade" : deal_subGrade,
    "employmentTitle": deal_employmentTitle,
    "employmentLength": deal_employmentLength,
    "homeOwnership": deal_homeOwnership,
    "annualIncome": deal_annualIncome,
    "verificationStatus": deal_verificationStatus,
    "issueDate": deal_issueDate,
    "purpose": deal_purpose,
    "postCode": deal_postCode,
    "regionCode": deal_regionCode,
    "dti": deal_dti,
    "delinquency_2years": deal_delinquency_2years,
    "ficoRangeLow": deal_ficoRangeLow,
    "ficoRangeHigh": deal_ficoRangeHigh,
    "openAcc": deal_openAcc,
    "pubRec": deal_pubRec,
    "pubRecBankruptcies": deal_pubRecBankruptcies,
    "revolBal" : deal_revolBal,
    "revolUtil": deal_revolUtil,
    "totalAcc" : deal_totalAcc,
    "initialListStatus" : deal_initialListStatus,
    "applicationType": deal_applicationType,
    "earliesCreditLine": deal_earliesCreditLine,
    "title": deal_title,
    "policyCode": deal_policyCode,
}


def train_label_encoders(data: pd.DataFrame,retrain = False):
    for col in data.columns:
        if not os.path.exists(pre_model_files[col]) or retrain:
            _train_label_encoder(data[col])

def _get_day(date):
    stamp = "2020-01-01"
    date1 = time.strptime(date, "%Y-%m-%d")
    date2 = time.strptime(stamp, "%Y-%m-%d")
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date2-date1).days

#%%
def add_features(data: pd.DataFrame):
    # 空值的数量
    data["null_sum"]=data.isnull().sum(axis=1)
    # 年、月
    data["earliesCreditLine_year"] = data["earliesCreditLine"].apply(lambda x: 2020 - np.int(x[-4:]))
    data["earliesCreditLine_month"] = data["earliesCreditLine"].apply(lambda x: month_dict[x[:3]])
    data["issueDate_year"] = data["issueDate"].apply(lambda x: np.int(x[:4]))
    data["issueDate_month"] = data["issueDate"].apply(lambda x: np.int(x[5:7]))
    data["issueData_all_month"] = data["issueDate_year"]*12 - data["earliesCreditLine_month"]
    data["issueDate_day"] = data["issueDate"].apply(lambda x: _get_day(x))
    data["issueDate_week_day"] = data["issueDate_day"].apply(lambda x: int(x % 7) + 1)
    
    # 贷款相关
    data["interestRate_term_rate"]=data["interestRate"]/(data["term"]+0.01)
    data["ave_loanAmnt_term_rate"] = data["loanAmnt"] / (data["term"]+0.01)
    data["loanAmnt_annualIncome"]=data["loanAmnt"]/ (data["annualIncome"]+0.01)

    # 还款-收入
    data['all_installment'] = data['installment'] * data['term']
    data["installment_annualIncome_rate"] = data["installment"]/(data["annualIncome"]+0.01)
    data["money_total"] = data["annualIncome"]*data["employmentLength"]
    data["annualIncome_employmentLength_rate"]=data["annualIncome"] / (data['employmentLength']+0.01)
    # 缴纳贷款剩余
    data["rest_money"] = data["annualIncome"] - data["ave_loanAmnt_term_rate"]

    # 历史信用情况  发放年-申请年
    data["earliesCreditLine_issueDate_diff"] = data["issueDate_year"] - (2020 - data["earliesCreditLine_year"])
    data["openAcc_TotalAcc_rate"]=data["openAcc"]/(data["totalAcc"]+0.01)
    data["Acc_diff"] = data['totalAcc'] - data['openAcc']
    data["pubRecBankruptcies_pubRec__rate"]=data["pubRecBankruptcies"]/(data["pubRec"]+0.01)
    data["Acc_diff"] = data['totalAcc'] - data['openAcc']
    data["pubRec_diff"]=data["pubRecBankruptcies"]-data["pubRec"]
    # 贷款利率/贷款时间
    
    # fico差值,均值
    data["fico_diff"]= data["ficoRangeHigh"] - data["ficoRangeLow"]
    data["fico_mean"] = (data['ficoRangeHigh'] + data['ficoRangeLow']) / 2
    # 贷款使用金额合计
    data["revolBal_total"]=data["revolBal"]*data["revolUtil"]*0.001
    data["rest_Revol"] = data["loanAmnt"] - data["revolBal"]
    
    data.drop(columns=["issueDate","earliesCreditLine"],inplace=True)
    return data


def add_count_features(data: pd.DataFrame, cols: list):
    enc = CountEncoder(cols=cols)
    ncol = enc.fit_transform(data[cols])
    return data.join(ncol,rsuffix="_enc")
    
def add_mean_features(data_train: pd.DataFrame, target ,cols: list, data_test):
    me = MeanEncoder(cols=cols, target_type = "classification")
    data_train = me.fit_transform(data_train,target)
    data_test = me.transform(data_test)
    return data_train, data_test


def _kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['isDefault'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['isDefault']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['isDefault'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


def add_target_features(train,test,target,target_encode_cols,kflod_num):
    train, test = _kfold_stats_feature(train, test, target_encode_cols, kflod_num)
    print('num1:target_encode train.shape', train.shape, test.shape)
    ### target encoding目标编码，回归场景相对来说做目标编码的选择更多，不仅可以做均值编码，还可以做标准差编码、中位数编码等
    enc_cols = []
    stats_default_dict = {
        'max': train['isDefault'].max(),
        'min': train['isDefault'].min(),
        'median': train['isDefault'].median(),
        'mean': train['isDefault'].mean(),
        'sum': train['isDefault'].sum(),
        'std': train['isDefault'].std(),
        'skew': train['isDefault'].skew(),
        'kurt': train['isDefault'].kurt(),
        'mad': train['isDefault'].mad()
    }
    ### 暂且选择这三种编码
    enc_stats = ['mean', 'skew', 'std' ]
    skf = KFold(n_splits=kflod_num, shuffle=True, random_state=seed)
    for f in tqdm(['postCode', 'regionCode', 'homeOwnership', 'employmentTitle','title']):
        enc_dict = {}
        for stat in enc_stats:
            enc_dict['{}_target_{}'.format(f, stat)] = stat
            train['{}_target_{}'.format(f, stat)] = 0
            test['{}_target_{}'.format(f, stat)] = 0
            enc_cols.append('{}_target_{}'.format(f, stat))
        for i, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
            trn_x, val_x = train.iloc[trn_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
            enc_df = trn_x.groupby(f, as_index=False)['isDefault'].agg(enc_dict)
            val_x = val_x[[f]].merge(enc_df, on=f, how='left')
            test_x = test[[f]].merge(enc_df, on=f, how='left')
            for stat in enc_stats:
                val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(
                    stats_default_dict[stat])
                test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(
                    stats_default_dict[stat])
                train.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
                test['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits

    print('num2:target_encode train.shape', train.shape, test.shape)

    train.drop(target_encode_cols, axis=1, inplace=True)
    test.drop(target_encode_cols, axis=1, inplace=True)
    print('输入数据维度：', train.shape, test.shape)
    return train, test

# def add_target_features(data_train: pd.DataFrame, target ,cols: list, data_test):
#     tc = TargetEncoder(cols = cols)
#     data_train = tc.fit_transform(data_train, target)
#     data_test = tc.transform(data_test)
#     return data_train, data_test

# %%
def load_origin_data(path):
    train_label = pd.read_csv(path + 'train.csv')['isDefault']
    data_train = pd.read_csv(path + 'train.csv')
    data_test = pd.read_csv(path + 'testA.csv')
    feats = [f for f in data_train.columns if f not in ['isDefault']]
    # train = train[feats]
    data_test = data_test[feats]
    print('train.shape', data_train.shape)
    print('test.shape', data_test.shape)
    return train_label, data_train, data_test


def data_preprocess():

    train_label, data_train, data_test  = load_origin_data(origin_data_path)
    data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True)

    numerical_fea = list(data_all.select_dtypes(exclude=['object']).columns)
    category_fea = list(filter(lambda x: x not in numerical_fea,list(data_all.columns)))
    label = 'isDefault'
    numerical_fea.remove(label)
    
    #按照中位数填充数值型特征
    data_all[numerical_fea] = data_all[numerical_fea].fillna(data_all[numerical_fea].median())
    #按照众数填充类别型特征
    data_all[category_fea] = data_all[category_fea].fillna(data_all[category_fea].mode())
    train_label_encoders(data_all[list(pre_model_files.keys())])
    data_all["employmentLength"] = deal_employmentLength(data_all["employmentLength"])
    data_all["grade"] = deal_grade(data_all["grade"])
    data_all["subGrade"] = deal_subGrade(data_all["subGrade"])
    data_all = add_features(data_all)
    data_all = add_count_features(data_all, cols=count_encode_cols)

    tmp_cols = ["loanAmnt","ficoRangeLow","ficoRangeHigh"]
    for col in tmp_cols:
        data_all[col]=func_map[col](data_all[col])
        
    data_train = data_all[~data_all['isDefault'].isnull()].copy()
    data_test = data_all[data_all['isDefault'].isnull()].copy()
    target = train_label

    data_train, data_test = add_mean_features(data_train,target,mean_encode_cols,data_test)
    # data_train, data_test = add_target_features(data_train,target,target_encode_cols,data_test)
    
    return data_train, target, data_test

#%%
def load_dataset():
    data_train, target, data_test  = data_preprocess()
    # target encoder
    target_encode_cols = ['postCode', 'regionCode', 'homeOwnership', 'employmentTitle','title']
    kflod_num = 5
    data_train, data_test = add_target_features(data_train,data_test,target,target_encode_cols,kflod_num)
    data_train.to_csv(data_train_path,index=None)
    data_test.to_csv(data_test_path,index=None)
    return data_train, data_test