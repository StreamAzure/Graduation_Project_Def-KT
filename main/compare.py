import sys
import os
import re
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# 获取文件名和要查找的字符串数量参数
if len(sys.argv) < 3:
    print("Usage: python plot_accuracy.py <num_of_strings1> <nums_range | <spectial_num1>>  ...")
    sys.exit(1)
num_of_strings1 = int(sys.argv[1])
baseline_filename = "baseline_20epoch.log"

num_list = []

if "-" in sys.argv[2]:
    for i in range(int(sys.argv[3].split("-")[0]), int(sys.argv[3].split("-")[1]) + 1):
        num_list.append(i)
else:
    for i in range(2, len(sys.argv)):
        num_list.append(int(sys.argv[i]))

# 获取当前目录下以".log"结尾的文件名列表
file_names = [file_name for file_name in os.listdir() if file_name.endswith(".log")]

# 筛选出符合数字范围的文件名
selected_file_names = []
for file_name in file_names:
    # 不以数字开头的文件名将被忽略
    if not file_name[0].isdigit():
        continue
    # 没有"-"的文件名单独处理
    if "-" not in file_name:
        file_num = int(file_name.split(".")[0])
        if file_num in num_list:
            selected_file_names.append(file_name)
        continue
    file_num = int(file_name.split("-")[0])
    if file_num in num_list:
        selected_file_names.append(file_name)

for file_name in selected_file_names:
    if not os.path.isfile(file_name):
        print(f"Error: {file_name} does not exist.")
        sys.exit(1)

# 打开baseline文件并读取内容
def read_file(filename, num_of_strings):
    with open(filename, "r") as f:
        content = f.read()

    # 记录准确率
    acc_list = []
    pattern_acc = r"Test accuracy:\s*(\d+\.\d+)%"
    matches = re.findall(pattern_acc, content)
    length = num_of_strings
    if len(matches) < num_of_strings:
        length = len(matches)
    for i in range(length):
        acc_list.append(float(matches[i]))

    # 取最后十轮的loss平均值作为收敛loss
    pattern_loss = r"Test loss:\s*(\d+\.\d+)"
    matches = re.findall(pattern_loss, content)
    avg_loss = round(sum([float(matches[i]) for i in range(-10, 0)]) / 10, 2)
    
    # 记录acc最大值
    max_acc = max(acc_list)

    # 取最后十轮的acc平均值作为收敛acc，保留两位小数
    avg_acc = round(sum(acc_list[-10:]) / 10, 2)

    return acc_list, max_acc, avg_acc, avg_loss

def read_file2(rounds):
    data_list = []
    max_acc = {}
    avg_acc = {}
    avg_loss = {}
    avg_DML_loss = {}
    for filename in selected_file_names:
        with open(filename, "r") as f:
            content = f.read()

        acc_list = []
        pattern = r"--- All clients' test acc: \s*(\d+\.\d+)%"
        matches = re.findall(pattern, content)
        length = rounds
        if len(matches) < rounds:
            length = len(matches)
        for i in range(length):
            acc_list.append(float(matches[i]))
        data_list.append(acc_list)

        # 记录acc最大值
        max_acc[filename] = max(acc_list)
        # 取最后十轮的acc平均值作为收敛acc，保留两位小数
        avg_acc[filename] = round(sum([float(matches[i]) for i in range(-10, 0)]) / 10, 2)
        # 取最后十轮的loss平均值作为收敛loss
        pattern = r"--- All clients' test loss: \s*(\d+\.\d+)"
        matches = re.findall(pattern, content)
        avg_loss[filename] = round(sum([float(matches[i]) for i in range(-10, 0)]) / 10, 2)
        # 取最后50条的DML loss平均值作为收敛DML loss
        pattern = r"--- DML_update_loss\(A model\) with Client:f(\d{7}): (\d+\.\d+)"
        matches = re.findall(pattern, content)
        avg_DML_loss[filename] = round(sum([float(matches[i][1]) for i in range(-50, 0)]) / 50, 2)

    return data_list, max_acc, avg_acc, avg_loss, avg_DML_loss

# 读取baseline的数据
baseline_acc_list, baseline_max_acc, baseline_avg_acc, baseline_avg_loss = read_file(baseline_filename, num_of_strings1)
# 读取多个文件的数据，每个文件的数据存储在一个列表中
data_list, max_acc, avg_acc, avg_loss, avg_DML_loss = read_file2(num_of_strings1)

# 根据acc_list1和data_list生成折线图
for i in range(len(data_list)):
    x = range(len(data_list[i]))
    y = data_list[i]
    plt.plot(x, y, label=f"{selected_file_names[i]}")

# 根据acc_list1生成折线图
x = range(len(baseline_acc_list))
y = baseline_acc_list
plt.plot(x, y, label=f"{baseline_filename}")

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('compare.png')

# note={
#     "17-.log": "Def-KT, local epoch = 5, DML epoch = 15",
#     "19.log": "Def-KT, local epoch = 20, DML epoch = 20",
#     "20.log": "Def-KT, local epoch = 10, DML epoch = 10",
# }

# 创建表格对象并指定表头
table = PrettyTable()
# table.field_names = ["num", "max acc", "avg acc", "avg loss(CE)", "DML loss(CE+KL)", "备注"]
table.field_names = ["num", "max acc", "avg acc", "avg loss(CE)", "DML loss(CE+KL)"]

# 添加baseline数据行
# table.add_row([baseline_filename, baseline_max_acc, baseline_avg_acc, baseline_avg_loss, "-", "baseline, local epoch = 20"])
table.add_row([baseline_filename, baseline_max_acc, baseline_avg_acc, baseline_avg_loss, "-"])
# 添加其他数据行
for file_name in selected_file_names:
    # table.add_row([file_name, max_acc[file_name], avg_acc[file_name], avg_loss[file_name], avg_DML_loss[file_name], note[file_name]])
    table.add_row([file_name, max_acc[file_name], avg_acc[file_name], avg_loss[file_name], avg_DML_loss[file_name]])
# 输出表格
print(table)
