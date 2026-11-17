import pandas as pd
import numpy as np

# 读取CSV文件
file_path = "results_scores_0.5_1new.csv"  # 请上传你的CSV文件
df = pd.read_csv(file_path)

# 确保数据格式正确
df = df.iloc[:, :4]  # 只取前四列
df.columns = ['Index', 'Value2', 'Value3', 'TrueClass']  # 设定列名

# 定义分类规则
# def classify(value2, value3):
#     if (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
#         return 0
#     elif abs(value2 - value3) <= 0.01:
#         return 1
#     elif (0.28 <= value2 <= 0.33 or value2 <= 0.05) and (0.65 <= value3 <= 0.75):
#         return 2
#     # elif (value3 < 0.28 or value3 > 0.32) and (value3 < (value2 - 0.01)):
#     #     return 3
#     # elif (value2 < 0.28 or value2 > 0.33) and (value2 < (value3 - 0.01)):
#     #     return 4
#     elif value2 < value3:
#         return 4
#     else:
#         return 3

# def classify(value2, value3):
#     if (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
#         return 0
#     elif abs(value2 - value3) <= 0.01:
#         return 1
#     elif (0.28 <= value2 <= 0.33 or value2 <= 0.05) and (0.65 <= value3 <= 0.75):
#         return 2
#     else:
#         return 3


def classify(value2, value3):
    if (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
        return 0
    elif abs(value2 - value3) <= 0.01:
        return 1
    elif (0.28 <= value2 <= 0.33 or value2 <= 0.05) and (0.65 <= value3 <= 0.75):
        return 2
    elif value2 < value3:
        return 2
    else:
        return 0
#
# 二分类
# def classify(value2, value3):
#     if abs(value2 - value3) <= 0.01:
#         return 1
#     else:
#         return 0


# 应用分类规则到第五列
df["PredictedClass"] = df.apply(lambda row: classify(row["Value2"], row["Value3"]), axis=1)

# 确保第五列填充了值
print(df.head())  # 打印前几行检查数据

# 计算类别准确率
accuracy_dict = {}
for category in [0, 1, 2]:
    total = len(df[df["TrueClass"] == category])
    correct = len(df[(df["TrueClass"] == category) & (df["PredictedClass"] == category)])
    accuracy_dict[category] = correct / total if total > 0 else 0

# 计算总体准确率
overall_accuracy = np.mean(list(accuracy_dict.values()))

# 输出分类准确率
print(f"类别0的准确率: {accuracy_dict[0]:.2%}")
print(f"类别1的准确率: {accuracy_dict[1]:.2%}")
print(f"类别2的准确率: {accuracy_dict[2]:.2%}")
# print(f"类别3的准确率: {accuracy_dict[3]:.2%}")
#(f"类别4的准确率: {accuracy_dict[4]:.2%}")
print(f"总体平均准确率: {overall_accuracy:.2%}")

output_file = "data123_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")

# import pandas as pd
# import numpy as np
#
# # 读取CSV文件
# file_path = "open_data.csv"  # 请上传你的CSV文件
# df = pd.read_csv(file_path)
#
# # 确保数据格式正确
# df = df.iloc[:, :4]  # 只取前四列
# df.columns = ['Index', 'Value2', 'Value3', 'TrueClass']  # 设定列名
#
# # 定义分类规则
# def classify(value2, value3):
#     if (value3 < 0.28 or value3 > 0.32) and (value3 < (value2 - 0.01)): #DM
#         return 3
#     elif (value2 < 0.28 or value2 > 0.33) and (value2 < (value3 - 0.01)): #GAN
#         return 4
#     elif (0.28 <= value3 <= 0.32 or value3 <= 0.05) and (0.65 <= value2 <= 0.75):
#         return 0
#     elif abs(value2 - value3) <= 0.01:
#         return 1
#     elif (0.28 <= value2 <= 0.33 or value2 <= 0.05) and (0.65 <= value3 <= 0.75):
#         return 2
#     else:
#         return -1  # 未分类数据
#
# # 应用分类规则到第五列
# df["PredictedClass"] = df.apply(lambda row: classify(row["Value2"], row["Value3"]), axis=1)
#
# # 确保第五列填充了值
# print(df.head())  # 打印前几行检查数据
#
# # 计算类别准确率
# accuracy_dict = {}
# for category in [0, 1]:
#     total = len(df[df["TrueClass"] == category])
#     correct = len(df[(df["TrueClass"] == category) & (df["PredictedClass"] == category)])
#     accuracy_dict[category] = correct / total if total > 0 else 0
#
# # 计算总体准确率
# overall_accuracy = np.mean(list(accuracy_dict.values()))
#
# # 输出分类准确率
# print(f"类别0的准确率: {accuracy_dict[0]:.2%}")
# print(f"类别1的准确率: {accuracy_dict[1]:.2%}")
# print(f"总体平均准确率: {overall_accuracy:.2%}")
#
# output_file = "data_with_predictions.csv"
# df.to_csv(output_file, index=False)
# print(f"分类结果已保存到 {output_file}")
