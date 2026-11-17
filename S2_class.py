import pandas as pd
import numpy as np

# S2: 真实 / 合成 二分类
# 说明：
# - 0 -> REAL
# - 1 -> SYNTHETIC

file_path = "results_scores_0.5_1new.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, :4]
df.columns = ['Index', 'Value2', 'Value3', 'TrueClass']


def classify(value2, value3):
    if abs(value2 - value3) <= 0.01:
        return 0
    else:
        return 1


df["PredictedClass"] = df.apply(lambda row: classify(row["Value2"], row["Value3"]), axis=1)

print(df.head())

accuracy_dict = {}
for category in [0, 1]:
    total = len(df[df["TrueClass"] == category])
    correct = len(df[(df["TrueClass"] == category) & (df["PredictedClass"] == category)])
    accuracy_dict[category] = correct / total if total > 0 else 0

overall_accuracy = np.mean(list(accuracy_dict.values()))

for category, name in zip([0, 1], ["REAL", "SYNTHETIC"]):
    print(f"类别{name}的准确率: {accuracy_dict[category]:.2%}")
print(f"总体平均准确率: {overall_accuracy:.2%}")

output_file = "data_S2_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")
