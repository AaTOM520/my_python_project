

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.tree import plot_tree

plt.rcParams['font.sans-serif']=['simHei']
#提取数据,拼接数据，重新划分数据
df1=pd.read_csv('test.csv')
df2=pd.read_csv('train.csv')
df3=pd.read_csv('gender_submission.csv')    #读取数据
values=df3['Survived']
df1.insert(loc=1,column='Survived',value=values)    #将df3中的‘Survived’这一特征放入df1第2列中
df=pd.concat([df2,df1])         #拼接成完整矩阵

#LabelEncoder标签编码，处理分类字符串（性别）(处理一维数据)
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])       #默认男1，女0

# OneHotEncoder，独热编码处理多分类字符串（船舱位置）（处理二维数据）
ohe=OneHotEncoder(sparse_output=False)
encoder=ohe.fit_transform(df[['Embarked']])#单括号取值为一维数据，sklearn作用于二维数据（双括号进行数据转化）
encoder_df=pd.DataFrame(encoder,columns=ohe.get_feature_names_out(['Embarked']),index=df.index)
df=pd.concat([df,encoder_df],axis=1)
df=df.drop('Embarked',axis=1)

# 剔除姓名（Name)，船票（Ticket),Cabin，
df=df.drop(['Name','Ticket','Cabin'],axis=1)
# print(df)
df=df.fillna(np.mean(df))           #用平均值填补缺失值
y=df['Survived']    #对象的重塑
# print(y)
x=df.drop('Survived',axis=1)     #除去目标target
# print(x)

#划分数据集与测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#使用VarianceThreshold进行数据清洗
variance_model=VarianceThreshold()
x_test_v=variance_model.fit_transform(x_train)
x_train_v=variance_model.transform(x_test)
#数据标准化
standard_model=StandardScaler()
x_test_standard=standard_model.fit_transform(x_train_v)
x_train_standard=standard_model.transform(x_test_v)
# print(x_test_standard)
# 超参数调优（随机森林进行预测）
param_grid={'n_estimators':[225,250,275],
            'max_depth':[None,5,10],
            'max_features':['sqrt','log2'],
            'min_samples_split':[1,2,5]}
grid_search=GridSearchCV(estimator=RandomForestClassifier(),
                         param_grid=param_grid,
                         cv=5,
                         scoring='accuracy',
                         n_jobs=-1)
grid_search.fit(x_train_standard,y_train)
print('最佳参数：',grid_search.best_params_)
print('交叉验证分数：',grid_search.best_score_)
#调用最佳参数进行预测
best_model=grid_search.best_estimator_
y_pred=best_model.predict(x_test_standard)
print('测试集准确率：',accuracy_score(y_test,y_pred))
print('分类报告：\n',classification_report(y_test,y_pred))
# 随机绘制决策树，进行可视化
tree2=best_model.estimators_[1]    #第二棵树
tree89=best_model.estimators_[88]  #第89棵树
plt.figure(figsize=(50,50))
plt.subplot(1,3,1)
plot_tree(tree2,filled=True,rounded=True,fontsize=8)
plt.title('第二棵树')
plt.subplot(1,3,2)
plot_tree(tree89,filled=True,rounded=True,fontsize=8)
plt.title('第89棵树')
plt.show()