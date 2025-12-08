import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # 改为分类器
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
from sklearn.feature_selection import VarianceThreshold

# ---------------------- 1. 读取数据（修正测试集标签处理） ----------------------
# 注意：test.csv本身无Survived列，gender_submission.csv是提交示例，若仅做预测可不用y_test
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# 若需评估，可将train_df拆分训练集和验证集，而非直接用gender_submission
# 此处暂时注释y_test相关，若有验证集需求可自行拆分
df = pd.read_csv('gender_submission.csv')
y_test = df['Survived']

target_col = 'Survived'  # 二分类目标变量
id_col = 'PassengerId'

# 拆分训练集特征和目标，测试集仅保留特征
X_train = train_df.drop([target_col, 'Name'], axis=1, errors='ignore')
y_train = train_df[target_col]
X_test = test_df.copy()

print(f"训练集特征形状: {X_train.shape}, 测试集特征形状: {X_test.shape}")

# ---------------------- 2. 数据清洗函数（保持不变，仅适配分类任务） ----------------------
def clean_train_data(data, target_col, id_col):
    """清洗训练集：删除重复值、高缺失率特征、异常值"""
    # 删除重复值（排除编号列）
    data = data.drop_duplicates(subset=[col for col in data.columns if col != id_col])
    # 剔除高缺失率特征（>50%，保留目标列和编号列）
    missing_rate = data.isnull().mean()
    drop_cols = missing_rate[missing_rate > 0.5].index.tolist()
    drop_cols = [col for col in drop_cols if col not in [target_col, id_col]]
    data = data.drop(columns=drop_cols)
    # 异常值处理（3σ原则，仅数值特征）
    numerical_temp = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_temp = [col for col in numerical_temp if col not in [target_col, id_col]]
    for col in numerical_temp:
        col_filled = data[col].fillna(data[col].median())
        z_score = stats.zscore(col_filled)
        valid_idx = np.abs(z_score) <= 3
        data = data.loc[valid_idx].reset_index(drop=True)
    return data, drop_cols

# 清洗训练集并同步处理测试集
train_df_clean, drop_cols = clean_train_data(train_df, target_col, id_col)
X_train = train_df_clean.drop(target_col, axis=1)
y_train = train_df_clean[target_col]
X_test = X_test.drop(columns=drop_cols, errors='ignore')

print(f"\n训练集清洗后形状: {X_train.shape}")
print(f"测试集同步删除列后形状: {X_test.shape}")

# ---------------------- 3. 特征分类与低方差剔除 ----------------------
# 特征分类（排除编号列）
numerical_cols = [col for col in X_train.select_dtypes(include=['int64', 'float64']).columns if col != id_col]
categorical_cols = [col for col in X_train.select_dtypes(include=['object', 'category']).columns if col != id_col]

# 数值特征低方差筛选
selected_num_cols = []
if numerical_cols:
    num_imputer = SimpleImputer(strategy='median')
    X_train_num_filled = num_imputer.fit_transform(X_train[numerical_cols])
    var_selector = VarianceThreshold(threshold=0.0)
    var_selector.fit(X_train_num_filled)
    selected_num_cols = [numerical_cols[i] for i in var_selector.get_support(indices=True)]
numerical_cols = selected_num_cols

print(f"\n低方差保留的数值特征: {numerical_cols}")
print(f"最终分类特征: {categorical_cols}")

# ---------------------- 4. 构建预处理管道 ----------------------
# 数值特征处理：中位数填充+标准化
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理：众数填充+独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 整合预处理（仅处理数值/分类特征，编号列保留）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'  # 保留编号列
)

# ---------------------- 5. 预处理数据（剔除编号列用于建模） ----------------------
# 训练集预处理
X_train_processed = preprocessor.fit_transform(X_train)
train_id = X_train[id_col].values
test_id = X_test[id_col].values

# 测试集预处理
X_test_processed = preprocessor.transform(X_test)

# 剔除编号列（编号列在remainder中，计算特征列数）
# 数值特征数 + 分类特征独热编码数 = 建模用特征数
num_num_cols = len(numerical_cols)
# 计算独热编码后的分类特征数
cat_ohe_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).shape[0]
total_feature_cols = num_num_cols + cat_ohe_cols

X_train_model = X_train_processed[:, :total_feature_cols]
X_test_model = X_test_processed[:, :total_feature_cols]

# ---------------------- 6. 随机森林分类器+超参数调优 ----------------------
# 超参数搜索空间（分类器适配）
search_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']  # 分类器的损失函数
}

# 构建模型（分类器+网格搜索，评分指标改为分类指标）
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=search_space,
    cv=5,
    scoring='accuracy',  # 分类任务用准确率
    n_jobs=-1
)

# 训练模型
grid_search.fit(X_train_model, y_train)

# 输出最佳模型和训练集评估
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train_model)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n最佳模型参数: {grid_search.best_params_}")
print(f"训练集准确率: {train_accuracy:.4f}")
print("训练集分类报告:\n", classification_report(y_train, y_train_pred))
print('训练集样本：',y_train)
# ---------------------- 7. 测试集预测+结果保存 ----------------------
# 预测类别（0/1），若要概率可用predict_proba，再取阈值0.5
y_pred = best_model.predict(X_test_model)
print(f"\n测试集预测结果示例: {y_pred[:10]}")
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
# 构建结果DataFrame并保存
result = pd.DataFrame({
    id_col: test_id,
    target_col: y_pred.astype(int)  # 确保为整数类型
})
result.to_csv('Titanic生还预测结果.csv', index=False)
print("\n预测结果已保存至：Titanic生还预测结果.csv")