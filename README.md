# 天气类型分类分析

![Weather](https://github.com/SolitaryEgo/Analysis-of-Weather-Type-Classification/blob/main/dataset-cover%20(1).jpg)

## 该数据集是人工生成的，用于模拟天气数据以完成分类任务。它包含各种与天气相关的特征，并将天气分为四种类型：雨天、晴天、多云和雪天。该数据集用于练习分类算法、数据预处理和异常值检测方法。


字段 | 说明 |
|----|---- |
Temperature (numeric) | 摄氏度为单位的温度，范围从极冷到极热。 |
Humidity (numeric) | 湿度百分比，包括高于 100% 的值以引入异常值。 |
Wind Speed (numeric) | 风速以公里/小时为单位，范围包括不切实际的高值。 |
Precipitation (%) (numeric) | 降水量百分比，包括异常值。 |
Cloud Cover (categorical) | 云量描述。 |
Atmospheric Pressure (numeric) | 大气压力，单位为 hPa，涵盖范围很广。 |
UV Index (numeric) | UV 指数，表示紫外线辐射的强度。 |
Season (categorical) | 记录数据的季节。 |
Visibility (km) (numeric) | 以公里为单位的能见度，包括非常低或非常高的值。 |
Location (categorical) | 记录数据的位置类型。 |
Weather Type (categorical) | 分类的目标变量，表示天气类型。 |


数据来源：>[kaggle](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data)
