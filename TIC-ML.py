# -*- coding: gbk -*-
from calendar import c
import datetime
from ensurepip import bootstrap
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error,roc_curve, auc
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from sklearn.metrics import brier_score_loss,make_scorer,precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve
import shap
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Neural Networks": MLPClassifier(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}
def bootstrap_confidence_interval(y_true, y_scores, y_pred, metric_func, n_bootstrap=1000):
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        y_true_resample, y_scores_resample, y_pred_resample = resample(y_true, y_scores, y_pred)
        bootstrap_sample = metric_func(y_true_resample, y_scores_resample, y_pred_resample)
        bootstrap_samples.append(bootstrap_sample)

    bootstrap_samples = np.array(bootstrap_samples)
    lower = np.percentile(bootstrap_samples, 2.5)
    upper = np.percentile(bootstrap_samples, 97.5)

    return lower, upper
def calculate_metrics(model, X_Data, y_Data, name):
    # 初始化一个字典来保存每个指标的所有样本值
    metrics_samples = {
        'auroc': [],
        'pr_auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f_score': []
    }

    # 进行1000次自助法抽样
    for _ in range(10):
        # 对数据进行自助法抽样
        X_resample, y_resample = resample(X_Data, y_Data)

        # 预测概率和标签
        y_scores = model.predict_proba(X_resample)[:, 1]
        y_pred = model.predict(X_resample)

        # 计算指标
        precision, recall, _ = metrics.precision_recall_curve(y_resample, y_scores)
        sort_idx = np.argsort(recall)
        precision = precision[sort_idx]
        recall = recall[sort_idx]
        metrics_samples['auroc'].append(metrics.roc_auc_score(y_resample, y_scores))
        metrics_samples['pr_auc'].append(metrics.auc(recall, precision))
        metrics_samples['accuracy'].append(metrics.accuracy_score(y_resample, y_pred))
        metrics_samples['sensitivity'].append(metrics.recall_score(y_resample, y_pred))
        metrics_samples['specificity'].append(metrics.recall_score(1-y_resample, 1-y_pred))
        metrics_samples['precision'].append(metrics.precision_score(y_resample, y_pred))
        metrics_samples['f_score'].append(metrics.f1_score(y_resample, y_pred))

    # 计算每个指标的95%置信区间
    metrics_dict = {}
    for metric, samples in metrics_samples.items():
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)
        metrics_dict[f'{metric}'] = np.mean(samples)
        metrics_dict[f'{metric}_lower'] = lower
        metrics_dict[f'{metric}_upper'] = upper

    # 保存为CSV文件
    df = pd.DataFrame(metrics_dict, index=[name])
    return df


def plot_precision_recall_curves(pr_curves, title, save_path):
    # 创建一个新的图形
    plt.figure(figsize=(8, 6))

    # 绘制每个模型的Precision-Recall曲线
    for i, (precision, recall, name,score) in enumerate(pr_curves):
        plt.plot(recall, precision, label=f'{name} ({score:.2f})')

    # 设置图形的范围和标签
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")

    # 保存图形
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curves(calibration_data, title, save_path):
    # 创建一个新的图形
    plt.figure(figsize=(8, 6))

    # 对于每个模型的校准曲线数据，绘制校准曲线
    for fraction_of_positives, mean_predicted_value, name,score in calibration_data:
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name} (Brier: {score:.2f})')

    # 绘制完美校准的线
    plt.plot([0, 1], [0, 1], '--', color='gray')

    # 设置图形的范围和标签
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    plt.title(title)
    plt.legend(loc="lower right")

    # 保存图形
    plt.savefig(save_path)
    plt.close()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号
labelCol='NEW-hypercoagulopathy'

# 加载数据
path=str(r".\Data\New-hypercoagulopathy.csv")
data = pd.read_csv(path,encoding='UTF-8')
#data = data[data['Pre-APTT'].notna() & data['Post-APTT'].notna()]
#data = data[data['source'] == 1]
feature_vars=pd.read_csv(str(r".\Data\特征变量1.csv"),encoding='UTF-8')
# 筛选出类型不是0的行
selected_rows = feature_vars[feature_vars['类型'] != 0]
# 获取这些行的列名
selected_columns = selected_rows['列名'].str.strip()
usedColumns=selected_columns.tolist()
usedColumns.append('Surgery start time')
usedColumns.append('source')
data=data[usedColumns]
    
# 检查重复的列名
duplicated_columns = data.columns[data.columns.duplicated()]

# 打印出重复的列名
print(f"Duplicated columns: {duplicated_columns.tolist()}")

# 删除重复的列
data = data.loc[:, ~data.columns.duplicated()]


for col in data.columns:
    if col!='Surgery start time':
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 计算每列的缺失值比例
missing_ratio = data.isnull().sum() / len(data)
# 找出缺失值比例大于0.3的列
columns_to_drop = missing_ratio[missing_ratio > 0.4].index
# 删除这些列
data = data.drop(columns_to_drop, axis=1)

#缺失值填充    
continuous_vars = []
categorical_vars = []
binary_vars = []
# 遍历每一列
for col in data.columns:
    if col=='Surgery start time':
        continue
    if data[col].nunique() < 4:
        binary_vars.append(col)
    # 如果唯一值数量小于10，我们假设它是分类的
    else:
        if data[col].nunique() < 10:
            categorical_vars.append(col)
        # 否则，我们假设它是连续的
        else:
            continuous_vars.append(col)
        
df=data
df[binary_vars] = df[binary_vars].fillna(0)
# 对于连续变量，我们可以直接使用KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[continuous_vars] = imputer.fit_transform(df[continuous_vars])
if len(categorical_vars)>0:
    # 对于分类变量，我们需要先将其转换为数值形式
    encoder = OrdinalEncoder()
    df[categorical_vars] = encoder.fit_transform(df[categorical_vars])

    # 然后我们可以使用KNNImputer
    df[categorical_vars] = imputer.fit_transform(df[categorical_vars])

    # 最后，我们可以将数值转换回原来的类别
    df[categorical_vars] = encoder.inverse_transform(df[categorical_vars])

allData = df
allData.to_csv(r'.\Data\output\filled_data1.csv', index=False,encoding='utf-8')

# num_rows = len(allData)

# allData['Surgery start time'] = pd.to_datetime(allData['Surgery start time'],format='mixed', errors='coerce')
# allData = allData.dropna(subset=['Surgery start time'])

# allData.sort_values('Surgery start time', inplace=True)

#allData=allData.sample(frac=1).reset_index(drop=True)
# data=allData[allData['source'] == 1]
# outSideData=allData[allData['source'] == 0]

# # 随机抽取3556行数据
# outSideData = allData.sample(n=3556, random_state=42)

# # 获取剩余的数据
# inSideData = allData.drop(outSideData.index)

# outSideData = outSideData.reset_index(drop=True)
# inSideData = inSideData.reset_index(drop=True)

outSideData= allData[allData['source'] == 1]
inSideData= allData[allData['source'] == 0]
#Pcount = sum(data[labelCol] == 1)
outSideData= outSideData.drop('source', axis=1)
outSideData= outSideData.drop('Surgery start time', axis=1)
inSideData= inSideData.drop('source', axis=1)
inSideData= inSideData.drop('Surgery start time', axis=1)

X = inSideData.drop(labelCol, axis=1)
y = inSideData[labelCol]

X_outTest=outSideData.drop(labelCol, axis=1)
y_outTest=outSideData[labelCol]
# outSideTestData=outSideData
# X_outTest=outSideTestData.drop(labelCol, axis=1)
# y_outTest=outSideTestData[labelCol]


# 划分训练集和测试集
X_train, X_insidetest, y_train, y_insidetest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= False)

#-------------start SMOTE----------------
# # 创建SMOTE对象
smote = SMOTE(sampling_strategy={1:3671},random_state=42)

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='not majority')

# # # 对训练集进行过采样
X_train, y_train = smote.fit_resample(X_train, y_train)

# undersample = RandomUnderSampler(sampling_strategy={1:1670})

# # # X_train, y_train = undersample.fit_resample(X_train, y_train)
# from imblearn.under_sampling import RandomUnderSampler

# # # 创建随机欠采样实例
# undersample = RandomUnderSampler(sampling_strategy='auto')

# # # 应用随机欠采样到训练数据集
# X_train, y_train = undersample.fit_resample(X_train, y_train)
#-------------end SMOTE----------------

# X_outTest=outSideData.drop(labelCol,axis=1)
# y_outTest=outSideData[labelCol]

# 计算每个特征与目标变量之间的皮尔逊相关系数
correlations = X_train.corrwith(y_train)

# 选择相关性最大的50个特征
top_50_features = correlations.abs().nlargest(50).index
X_top_50 = X[top_50_features]

# 使用LASSO回归进行特征选择
lasso = Lasso(alpha=0.01)
lasso.fit(X_top_50, y)

# 使用SelectFromModel选择特征
model = SelectFromModel(lasso, max_features=25, prefit=True)
selected_features = X_top_50.columns[model.get_support()]

# 打印选择的特征
print("Selected features:", selected_features)


# 使用选择的特征转换数据
X_train = X_train[selected_features]
X_insidetest= X_insidetest[selected_features]
X_outTest = X_outTest[selected_features]

# 创建标准化器
scaler = StandardScaler()
X_train_original = X_train.copy()
# 对训练数据进行标准化
X_train = scaler.fit_transform(X_train)

# 使用相同的标准化器对测试数据进行标准化
X_insidetest=scaler.transform(X_insidetest)
X_outTest =scaler.transform(X_outTest)

# 假设 X 是你的特征数据
X_train = np.array(X_train)

# 计算Z-score
z_scores = np.abs(stats.zscore(X_train))

# 定义一个阈值，通常我们选择3，这意味着所有Z-score大于3的点都被认为是异常值
threshold = 3

# 获取异常值的位置
outliers = np.where(z_scores > threshold)

# 处理异常值，这里我们选择将它们替换为阈值
X_train[outliers] = threshold

# 定义要训练的模型列表
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Neural Networks": MLPClassifier(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Neural Networks": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# 优化每个模型的参数 --this part cost a lot of time, should skip when debug.
best_models = {}
for name, model in models.items():
    print(f"Optimizing {name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")


feature_importances = pd.DataFrame(index=selected_features)

# 创建一个空的DataFrame来保存结果
train_95CI = pd.DataFrame()
test_95CI = pd.DataFrame()
insidetest_95CI = pd.DataFrame()
outside_95CI = pd.DataFrame()

# 初始化一个列表来保存ROC曲线数据
roc_data = []
roc_data1 = []
roc_data2 = []
roc_data3 = []

# 定义评分参数
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
    'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
}

# 创建一个空的DataFrame来保存结果
results = pd.DataFrame()
# 创建一个空的DataFrame来保存结果
bootstrapResults = pd.DataFrame()
bootstrapResults1 = pd.DataFrame()
# 创建一个空列表来保存每个模型的Precision-Recall曲线的数据
pr_curves = []
pr_curves1 = []
pr_curves2 = []
pr_curves3 = []
# 创建一个空列表来保存每个模型的校准曲线的数据
calibration_data = []
calibration_data1 = []
calibration_data2 = []
calibration_data3 = []

# 对于每个模型，训练并评估它
for name, model in best_models.items():
    
    ################## start cross_validate
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)  # 5折交叉验证
    # 计算平均得分并保存到DataFrame
    for key in scores:
        if key.startswith('test_'):
            for i, score in enumerate(scores[key]):
                results.loc[f'{name}_{i}', f'{key}'] = score
    ################## start bootstrap
    # 创建一个空的字典来保存每次Bootstrap的得分
    bootstrap_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for i in range(3):  # 进行100次Bootstrap
        # 生成Bootstrap样本
        X_resample, y_resample = resample(X, y)
        # 训练模型并计算得分
        model.fit(X_resample, y_resample)
        y_pred = model.predict(X)
        bootstrap_scores['accuracy'].append(accuracy_score(y, y_pred))
        bootstrap_scores['precision'].append(precision_score(y, y_pred, average='weighted'))
        bootstrap_scores['recall'].append(recall_score(y, y_pred, average='weighted'))
        bootstrap_scores['f1'].append(f1_score(y, y_pred, average='weighted'))
    # 计算平均得分并保存到DataFrame
    for metric in bootstrap_scores.keys():
        mean = np.mean(bootstrap_scores[metric])
        lower = np.percentile(bootstrap_scores[metric], 2.5)  # 计算2.5百分位数，即置信区间的下限
        
        upper = np.percentile(bootstrap_scores[metric], 97.5)  # 计算97.5百分位数，即置信区间的上限
        bootstrapResults.loc[name, f'{metric} Mean'] = mean
        bootstrapResults.loc[name, f'{metric} Lower 95% CI'] = lower
        bootstrapResults.loc[name, f'{metric} Upper 95% CI'] = upper

    model.fit(X_train, y_train)
    train_95CIitem = calculate_metrics(model, X_train, y_train, name)
    train_95CI = pd.concat([train_95CI,train_95CIitem]) 
    insidetest_95CIitem=calculate_metrics(model, X_insidetest, y_insidetest, name)
    insidetest_95CI = pd.concat([insidetest_95CI,insidetest_95CIitem])  
    outside_95CIitem=calculate_metrics(model, X_outTest, y_outTest, name)
    outside_95CI = pd.concat([outside_95CI,outside_95CIitem])
    

    # 特征重要性pfi
    if hasattr(model, 'feature_importances_'):
        # 对于随机森林
        importances = model.feature_importances_
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 对于逻辑回归和线性SVM
        # importances = np.abs(model.coef_[0])  # 取绝对值，因为系数可以为负
        importances = model.coef_[0]  # 取绝对值，因为系数可以为负
        feature_importances[name] = model.coef_[0]

    # 评估
    predictions = model.predict(X_outTest)
    predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_outTest, predictions)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_outTest)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_outTest, y_score)
    roc_auc = auc(fpr, tpr)
    # 假设 y_outTest 和 y_score 是你要保存的数据
    # 首先，我们将它们转换为DataFrame
    ddf = pd.DataFrame({
        'y_outTest': y_outTest,
        'y_score': y_score
    })

    # 然后，我们可以使用 to_csv 方法将 DataFrame 保存为 CSV 文件
    ddf.to_csv('y_outTest-score.csv', index=False)
    # 保存ROC曲线数据
    roc_data.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线
    threshold = 0.5  # 设置阈值为0.5
    predictions = (y_score > threshold).astype(int)  # 将预测概率值大于阈值的设置为1，小于等于阈值的设置为0
    precisionscore = precision_score(y_outTest, predictions)
    precision, recall, _ = precision_recall_curve(y_outTest, y_score)
    brier_score = brier_score_loss(y_outTest, y_score)
    # 将Precision-Recall曲线的数据添加到列表中
    pr_curves.append((precision, recall, name, precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_outTest, y_score, n_bins=5)
    calibration_data.append((fraction_of_positives, mean_predicted_value, name,brier_score))

    ####################### insidetest
    predictions = model.predict(X_insidetest)
    accuracy = accuracy_score(y_insidetest, predictions)
    print(f'{name} insidetestData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_insidetest)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_insidetest, y_score)
    roc_auc = auc(fpr, tpr)

    # 保存ROC曲线数据
    roc_data1.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线
    threshold = 0.5  # 设置阈值为0.5
    predictions = (y_score > threshold).astype(int)  # 将预测概率值大于阈值的设置为1，小于等于阈值的设置为0
    precisionscore = precision_score(y_insidetest, predictions)
    precision, recall, _ = precision_recall_curve(y_insidetest, y_score)
    brier_score = brier_score_loss(y_insidetest, y_score)
    # 将Precision-Recall曲线的数据添加到列表中
    pr_curves1.append((precision, recall, name,precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_insidetest, y_score, n_bins=5)
    calibration_data1.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    #################### train
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    print(f'{name} trainData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_train)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)

    # 保存ROC曲线数据
    roc_data3.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线
    threshold = 0.5  # 设置阈值为0.5
    predictions = (y_score > threshold).astype(int)  # 将预测概率值大于阈值的设置为1，小于等于阈值的设置为0
    precisionscore = precision_score(y_train, predictions)
    precision, recall, _ = precision_recall_curve(y_train, y_score)
    brier_score = brier_score_loss(y_train, y_score)
    # 将Precision-Recall曲线的数据添加到列表中
    pr_curves3.append((precision, recall, name,precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, y_score, n_bins=5)
    calibration_data3.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    # 获取当前日期和时间
    now = datetime.datetime.now()

    # # 将日期和时间格式化为字符串
    # now_str = now.strftime('%Y%m%d_%H%M')    
    # # 计算SHAP值
    # # 使用shap.sample对数据进行采样
    # X_sample = shap.sample(X_train, nsamples=500)

    # # 根据模型类型选择解释器
    # if name in ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost"]:
    #     explainer = shap.TreeExplainer(model)
    # elif name == "Neural Networks":
    #     try:
    #         explainer = shap.DeepExplainer(model, X_train)
    #     except Exception:
    #         try:
    #             shap.KernelExplainer(model.predict, X_sample)
    #         except Exception:
    #             continue
    # elif name in ["Logistic Regression"]:
    #     explainer = shap.LinearExplainer(model, X_train)
    # else:
    #     explainer = shap.KernelExplainer(model.predict, X_sample)


    # shap_values = explainer.shap_values(X_sample)

    # plt.figure()
    # # 绘制SHAP值
    # shap.summary_plot(shap_values, X_sample, max_display=15, feature_names=selected_features,show=False)
    # plt.savefig(f'shap/{name}-shap_plot-{now_str}.png',dpi=700)
    # plt.close()
    # plt.clf()

# 获取当前日期和时间
now = datetime.datetime.now()

# 将日期和时间格式化为字符串
now_str = now.strftime('%Y%m%d_%H%M%S')    
# 保存结果到CSV文件
results.to_csv(f'cross_validation_results_{now_str}.csv')
bootstrapResults.to_csv(f'bootstrap_results_{now_str}.csv')


train_95CI.to_csv(f'train_95CI_results_{now_str}.csv')
insidetest_95CI.to_csv(f'insidetest_95CI_results_{now_str}.csv')
test_95CI.to_csv(f'test_95CI_results_{now_str}.csv')
outside_95CI.to_csv(f'outside_95CI_results_{now_str}.csv')
   
# 过滤出重要性大于0.05的特征
threshold = 0.0005
important_features = feature_importances[feature_importances > threshold].dropna(how='all')

# 保存到CSV文件
important_features.to_csv(f'important_features_{now_str}.csv', encoding='UTF-8')    
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Outside Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'Outside roc_curves_{now_str}.png')
#plt.show()

# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data1:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Inside Test Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'Inside Test roc_curves_{now_str}.png')
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data3:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'Train Data roc_curves_{now_str}.png')
#plt.show()
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data2:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'Test Data roc_curves_{now_str}.png')
#plt.show()
plot_calibration_curves(calibration_data, 'Calibration curves', f'outside_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data1, 'InsideTest Data Calibration curves', f'insidetest_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data3, 'Train Data Calibration curves', f'train_data_calibration_curves_{now_str}.png')

plot_precision_recall_curves(pr_curves, 'Precision-Recall curve', f'outside_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves1, 'Precision-Recall curve', f'insidetest_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves3, 'Precision-Recall curve', f'train_data_precision_recall_curve_{now_str}.png')



