import pandas as pd
import numpy as np

# 读取CSV文件
file_path = "results_scores_0.5_1new.csv"
df = pd.read_csv(file_path)

# 确保数据格式正确
df = df.iloc[:, :4]
df.columns = ['Index', 'Value2', 'Value3', 'TrueClass']

# 3class.py
# 类别说明：0 / 1 / 2 / 3
def classify(value2, value3):
    if (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
        return 0
    elif abs(value2 - value3) <= 0.01:
        return 1
    elif (0.28 <= value2 <= 0.33 or value2 <= 0.05) and (0.65 <= value3 <= 0.75):
        return 2
    else:
        return 3


df["PredictedClass"] = df.apply(lambda row: classify(row["Value2"], row["Value3"]), axis=1)

print(df.head())

accuracy_dict = {}
for category in [0, 1, 2, 3]:
    total = len(df[df["TrueClass"] == category])
    correct = len(df[(df["TrueClass"] == category) & (df["PredictedClass"] == category)])
    accuracy_dict[category] = correct / total if total > 0 else 0

overall_accuracy = np.mean(list(accuracy_dict.values()))

for category in [0, 1, 2, 3]:
    print(f"类别{category}的准确率: {accuracy_dict[category]:.2%}")
print(f"总体平均准确率: {overall_accuracy:.2%}")

output_file = "data3class_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")
