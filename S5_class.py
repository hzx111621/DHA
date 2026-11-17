import pandas as pd
import numpy as np

# S5: 开放集四分类
# 说明：
# - 0 -> REAL
# - 1 -> GAN
# - 2 -> DM
# - 3 -> UNSEEN

file_path = "results_scores_0.5_1new.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, :4]
df.columns = ['Index', 'Value2', 'Value3', 'TrueClass']


def classify(value2, value3):
    if abs(value2 - value3) <= 0.01:
        return 0
    elif (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
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

for category, name in zip([0, 1, 2, 3], ["REAL", "GAN", "DM", "UNSEEN"]):
    print(f"类别{name}的准确率: {accuracy_dict[category]:.2%}")
print(f"总体平均准确率: {overall_accuracy:.2%}")

output_file = "data_S5_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")
