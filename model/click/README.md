### 点击预测模型
Classification
Regression
分离器：支持k-fold、 比率ratio 、 leave-one-out分离数据集
模型：推荐模型基于协同过滤算法，包括矩阵分解、基于临接的方法、Slope One、Co-Clustering2
评估：可使用RMSE、 MAE来评分，包括准确率Precision、召回率Recall、归一化折损累积增益NDCG、MAP、MRR、AUC
参数搜寻：使用方式网格搜索grid search 或 随机搜索random search寻找最佳超参数
持久化：保存模型或加载模型