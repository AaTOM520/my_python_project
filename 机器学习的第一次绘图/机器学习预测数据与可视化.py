import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import plot_tree

#配置中文字幕
plt.rcParams['font.sans-serif']=['simHei']
#数据加载
iris=load_iris()
# print(iris.data)
# print(iris.target)
# print(iris.feature_names)
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#标准化数据
model1=StandardScaler()
x_train1=model1.fit_transform(x_train)
x_test1=model1.transform(x_test)
#移除无关特征（数据质量差）(VarianceThreshold)
model2=VarianceThreshold(threshold=0)
x_test2=model2.fit_transform(x_test1)
x_train2=model2.transform(x_train1)
# print(x_test2,len(x_test2))
#选择3个最好的特征(SelectKBest)
model3=SelectKBest(score_func=f_classif,k=2)
x_train3=model3.fit_transform(x_train2,y_train)
x_test3=model3.transform(x_test2)
# print(x_test3)
#超参数调优（随机森林进行预测）
param_grid={
    'n_estimators':[50,100,200],    #树的数量
    'max_depth':[None,5,10],   #树的深度
    'max_features':['sqrt','log2'],     #分裂考虑最大特征数
    'min_samples_split':[2,5,10]        #节点分裂最小样本数
}
model4=GridSearchCV(estimator=RandomForestClassifier(),
                    param_grid=param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                    )
model4.fit(x_train3,y_train)
print('最佳参数:',model4.best_params_)
print('最佳交叉验证分数：',model4.best_score_)
#调用最佳模型进行预测
best_model=model4.best_estimator_
y_pred_best=best_model.predict(x_test3)
print('测试集准确率：',accuracy_score(y_test,y_pred_best))
print('分类报告：\n',classification_report(y_test,y_pred_best,target_names=iris.target_names))
#绘制决策树
tree1=best_model.estimators_[0]     #选择森林中的第1棵树
tree6=best_model.estimators_[5]     #选择森林中的第6棵树
tree10=best_model.estimators_[9]       #选择森林中的第10棵树
feature_names_cn=['花瓣长度','花瓣宽度']
class_names_cn=['山鸢尾','变色鸢尾','维吉尼亚鸢尾']
plt.figure(figsize=(30,10))
#第一棵树
plt.subplot(1,3,1)
plot_tree(
    tree1,
    filled=True,
    feature_names=feature_names_cn,
    class_names=class_names_cn,
    rounded=True,
    fontsize=8
)
plt.title('鸢尾花随机森林第1棵树')
#第六颗树
plt.subplot(1,3,2)
plot_tree(
    tree6,
    filled=True,
    feature_names=feature_names_cn,
    class_names=class_names_cn,
    rounded=True,
    fontsize=8
)
plt.title('鸢尾花随机森林第6棵树')
#第10棵树
plt.subplot(1,3,3)
plot_tree(
    tree10,
    filled=True,
    feature_names=feature_names_cn,
    class_names=class_names_cn,
    rounded=True,
    fontsize=8
)
plt.title('鸢尾花随机森林第10棵树')

plt.show()