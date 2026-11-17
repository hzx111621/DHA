import csv


def count_close_values(csv_file):
    count = 0

    # 打开并读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # 遍历每一行数据
        for row in reader:
            # 确保行中有足够的列
            if len(row) >= 3:  # 至少有四列
                try:
                    # 获取第三列和第四列的数据，并转换为浮点数
                    col3 = float(row[1])  # 第三列（索引从0开始）
                    col4 = float(row[2])  # 第四列

                    # 计算绝对值之差
                    diff = abs(col3 - col4)

                    # 判断是否在0.05之间
                    if diff <= 0.01:
                        count += 1
                except ValueError:
                    # 如果转换为浮点数失败，跳过该行
                    continue

    # 输出符合条件的数量
    print(f"绝对值之差在0.05之间的数量: {count}")


# 示例调用
csv_file_path = "resultsREAL.csv"  # 替换为你的CSV文件路径
count_close_values(csv_file_path)


# import csv
#
#
# def count_special_conditions(csv_file):
#     count = 0
#
#     with open(csv_file, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#
#         for row in reader:
#             # 确保至少有3列（索引0,1,2）
#             if len(row) >= 3:
#                 try:
#                     col2 = float(row[1])  # 第二列
#                     col3 = float(row[2])  # 第三列
#
#                     # 条件判断
#                     condition1 = (col3 < (col2 - 0.01))  # 第三列比第二列小至少0.05
#                     condition2 = (col3 <= 0.28) or (col3 >= 0.32)  # 值范围条件
#
#                     if condition1 and condition2:
#                         count += 1
#
#                 except ValueError:
#                     # 跳过无法转换为浮点数的行
#                     continue
#
#     print(f"满足条件的数量: {count}")
#
#
# # 示例调用
# csv_file_path = "resultsGLIDE.csv"  # 替换为你的CSV文件路径
# count_special_conditions(csv_file_path)