import matplotlib as mpl
import matplotlib.pyplot as plt

def plt_config():
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    ax = plt.gca()  # 获取当前轴对象

    # 设置刻度线的位置
    ax.tick_params(axis='both', which='both', direction='in', length=5, width=1)

    # 开启网格线
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # 设置图例线条的长度
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_linelength(10)

    return ax

# 恢复 matplotlib 的默认设置
mpl.rcdefaults()

# 调用 plt_config() 函数来设置字体、刻度线、网格线和图例线条长度
plt_config()

# 绘制图形
plt.plot([1, 2, 3], [4, 5, 6], label='Line 1')
plt.plot([1, 2, 3], [2, 4, 6], label='Line 2')
plt.legend()
plt.savefig('test.png')