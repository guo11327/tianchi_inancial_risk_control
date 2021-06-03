# 零基础入门金融风控-贷款违约预测

## 背景介绍
赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。通过这道赛题来引导大家了解金融风控中的一些业务背景，解决实际问题，帮助竞赛新人进行自我练习、自我提高。
本项目为本人练习项目，仅使用XGBoost模型进行预测。

[项目链接](https://tianchi.aliyun.com/competition/entrance/531830/introduction)

## 代码结构
- data
    - dataset 生成的数据集路径
    - origin_data 原始数据路径
    - result 预测结果路径
- model
    - xgboost 模型保存地址
    - xgbost.py 模型文件

- feature.py 特征处理文件
- main.py 程序主入口
- myconfig.py 项目配置文件
- preprocess.py 数据预处理文件
- utils.py  若干函数工具

## 运行方式
- 安装相关依赖包
- 下载项目数据，并放置在  data/origin_data/ 目录下
    - train.csv
    - testA.csv
    - sample_submit.csv

- 运行主文件
```
python main.py
```

