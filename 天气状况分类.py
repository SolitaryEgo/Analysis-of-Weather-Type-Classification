import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('./weather_classification_data.csv')

print(df.head())
print(df.duplicated().sum())
print(df.isna().sum())

df = df[(df['Temperature'] <= 60) & (df['Humidity'] <= 100) & (df['Precipitation (%)'] <= 100) & (
            df['Atmospheric Pressure'] <= 1050)]

print(df.describe(include='all'))

for i in ['Cloud Cover', 'Season', 'Weather Type', 'Location']:
    print(df[i].value_counts().reset_index(name='count'))

corr_df = df.select_dtypes(include=[np.number]).corr().round(2)

fig = px.imshow(corr_df, text_auto=True, aspect='auto')
fig.update_xaxes(side='top')
fig.update_layout(width=1000, height=500)
fig.show()

ndf = df.copy()
cat_list = ['Cloud Cover', 'Season', 'Location']
ndf = pd.get_dummies(ndf, columns=cat_list)

x = ndf.drop('Weather Type', axis=1)
y = ndf['Weather Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 选择模型并训练
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
print(f'模型预测准确度：{accuracy_score(y_test, y_pred) * 100:.2f}%')

# 使用训练好的模型输出特征重要性
feature_importances = rf_model.feature_importances_ * 100
f_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})
f_importance_df = f_importance_df.sort_values(by='Importance', ascending=False, ignore_index=True)
print(f_importance_df)

# 创建SelectFromModel创建对象，并使用随机森林分类器作为基础估计器。threshold={'median'|'mean'}表示选择特征重要性高于中位数/平均数的特征。
selector = SelectFromModel(RandomForestClassifier(), threshold='median')
selector = SelectFromModel(RandomForestClassifier(), threshold='median')
selector.fit(x_train, y_train)

# 使用transform方法从训练集中选择特征
x_train_selector = selector.transform(x_train)
x_test_selector = selector.transform(x_test)

# 使用选择的特征重新训练随机森林分类器
rf_model_selected = RandomForestClassifier()
rf_model_selected.fit(x_train_selector, y_train)

pred_sel = rf_model_selected.predict(x_test_selector)

# 使用训练好的模型输出特征重要性
feature_importances_sel = rf_model_selected.feature_importances_ * 100
selected_feature_names = x.columns[selector.get_support()]
f_sel_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances_sel})
f_sel_df = f_sel_df.sort_values(by='Importance', ascending=False, ignore_index=True)
print(f_sel_df)

print(f'模型预测准确度：{accuracy_score(y_test, pred_sel) * 100:.2f}%')

# 交叉验证
# N折交叉验证（5-fold cross-validation）的意思是将数据集分成N个相等的部分（称为“折”），然后进行N次迭代，在每次迭代中，选择其中一个部分作为测试集，其余N-1个部分组成训练集。这样，每个部分都会被用作一次测试集，而其余部分则作为训练集。
# 使用cross_val_score函数进行5折交叉验证，scoring='accuracy'表示使用准确率作为评估指标。
cv_scores = cross_val_score(rf_model_selected, x_train_selector, y_train, cv=5, scoring='accuracy')
mean_av_scores = cv_scores.mean() * 100
print(f'平均交叉验证准确度:{mean_av_scores:.2f}%')

precision = precision_score(y_test, pred_sel, average='weighted') * 100
print(f'模型预测准确度:{precision:.2f}%')

recall = recall_score(y_test, pred_sel, average='weighted') * 100
print(f'模型召回准确度:{recall:.2f}%')

print()
report = classification_report(y_test, pred_sel)
print(report)

# Cloudy：
# Precision (精确率): 0.94，意味着模型预测为多云的样本中有94%实际上是多云。
# Recall (召回率): 0.92，意味着实际多云的样本中有92%被模型正确识别为多云。
# F1-Score (F1分数): 0.93，是精确率和召回率的加权平均值。
# Support (支持度): 635，即测试集中实际多云的样本总数。


# https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data
