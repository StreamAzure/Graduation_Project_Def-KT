import sys
import os
import re
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

rounds = int(sys.argv[1])

def get_file_name():
    # 获取文件名和要查找的字符串数量参数
    if len(sys.argv) < 2:
        print("Usage: python compare.py file1 file2 file3 ...")
        sys.exit()

    file_list = []
    for i in range(2, len(sys.argv)):
        file_list.append(sys.argv[i])

    for file_name in file_list:
        if not os.path.isfile(file_name):
            print(f"Error: {file_name} does not exist.")
            sys.exit(1)
    
    return file_list

# 打开baseline文件并读取内容
def read_baseline_file(filename, num_of_strings):
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

    res = [acc_list, max_acc, avg_acc, avg_loss]

    return res

def read_single_DML_file(rounds, kt_file):
    with open(kt_file, "r") as f:
        content = f.read()
    acc_list = []
    pattern = r"--- All clients' test acc: \s*(\d+\.\d+)%"
    matches = re.findall(pattern, content)
    length = rounds
    if len(matches) < rounds:
        length = len(matches)
    for i in range(length):
        acc_list.append(float(matches[i]))

    # 取最后十轮的loss平均值作为收敛loss
    pattern_loss = r"--- All clients' test loss: \s*(\d+\.\d+)"
    matches = re.findall(pattern_loss, content)
    # avg_loss = round(sum([float(matches[i]) for i in range(-10, 0)]) / 10, 2)
    avg_loss = 1
    # 记录acc最大值
    max_acc = max(acc_list)
    # 取最后十轮的acc平均值作为收敛acc，保留两位小数
    avg_acc = round(sum(acc_list[-10:]) / 10, 2)
    # 取最后50条的DML loss平均值作为收敛DML loss
    pattern = r"--- DML_update_loss\(A model\) with Client:(\w+): (\d+\.\d+)"
    matches = re.findall(pattern, content)
    # avg_DML_loss = round(sum([float(matches[i][1]) for i in range(-40, 0)]) / 40, 2)
    avg_DML_loss=1
    res = [acc_list, max_acc, avg_acc, avg_loss, avg_DML_loss]
    return res
    
def read_DML_file(rounds, selected_file_names):
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
        pattern = r"--- DML_update_loss\(A model\) with Client:(\w+): (\d+\.\d+)"
        matches = re.findall(pattern, content)
        avg_DML_loss[filename] = round(sum([float(matches[i][1]) for i in range(-40, 0)]) / 40, 2)

    res = [data_list, max_acc, avg_acc, avg_loss, avg_DML_loss]

    return res

def plt_config():
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    ax = plt.gca()  # 获取当前轴对象

    # 设置刻度线的位置
    ax.tick_params(axis='both', which='both', direction='in', length=5, width=1)

    # 开启网格线
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

    # 设置图例线条的长度
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        # handle.set_linewidth(1)
        pass

    # ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))

    return ax

# 定义平滑函数
def moving_average(x, y, window_size):
    window = np.ones(int(window_size))/float(window_size)
    y_smooth = np.convolve(y, window, 'same')
    return y_smooth

def draw_plot(rounds, kt_file, baseline_file):    
    res_list = []
    baseline_acc_list = []
    kt_acc_list =[]
    # 读取baseline的数据
    res_list = read_baseline_file(baseline_file, rounds)
    baseline_acc_list = res_list[0]
    # 读取kt的数据
    res_list = read_single_DML_file(rounds, kt_file)
    kt_acc_list = res_list[0]
    
    x = range(len(kt_acc_list))
    y = kt_acc_list
    y_smooth = lowess(y, x, frac=0.1, return_sorted=False)
    plt_config()
    plt.plot(x, y_smooth, label=f"Def-KT")

    x = range(len(baseline_acc_list))
    y = baseline_acc_list
    y_smooth = lowess(y, x, frac=0.1, return_sorted=False)
    plt_config()
    plt.plot(x, y_smooth, label=f"FedAvg")

    plt.legend()

    plt.title("CIFAR-100, non-IID(ξ=80)")
    plt.xlabel("Number of rounds")
    plt.ylabel("Accuracy")
    plt.savefig('compare.png')

def printTable(kt_file, baseline_file):
    res_list = []
    res_list = read_baseline_file(baseline_file, rounds)
    baseline_max_acc, baseline_avg_acc, baseline_avg_loss = res_list[1], res_list[2], res_list[3]

    res_list = read_single_DML_file(rounds, kt_file)
    max_acc, avg_acc, avg_loss, avg_DML_loss = res_list[1], res_list[2], res_list[3], res_list[4]

    # 创建表格对象并指定表头
    table = PrettyTable()
    # table.field_names = ["num", "max acc", "avg acc", "avg loss(CE)", "DML loss(CE+KL)", "备注"]
    table.field_names = ["num", "max acc", "avg acc", "avg loss(CE)", "DML loss(CE+KL)"]

    # 添加baseline数据行
    # table.add_row([baseline_file, baseline_max_acc, baseline_avg_acc, baseline_avg_loss, "-", "baseline, local epoch = 20"])
    table.add_row([baseline_file, baseline_max_acc, baseline_avg_acc, baseline_avg_loss, "-"])
    # 添加其他数据行
    table.add_row([kt_file, max_acc, avg_acc, avg_loss, avg_DML_loss])
    # 输出表格
    print(table)

filenames = get_file_name()
baseline_file = ""
kt_file = ""
for file in filenames:
    if "baseline" in file:
        baseline_file = file
        break
for file in filenames:
    if "kt" in file:
        kt_file= file
        break
draw_plot(rounds, kt_file, baseline_file)
printTable(kt_file, baseline_file)