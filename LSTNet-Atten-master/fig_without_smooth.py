import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

i = 1500
# data = pd.read_csv('C:\\Users\\CaiH\\Desktop\\work2-LST\\pred.csv', header=0)#注意数据的格式，确保都是小数
data = pd.read_csv('C:\\Users\\KB9forever\\Desktop\\毕业设计\\LSTNet代码\\pred-all.csv', header=0)#注意数据的格式，确保都是小数
# C:\Users\24207\Desktop\代码\pred-all.csv
# C:\Users\KB9forever\Desktop\毕业设计\LSTNet代码\pred-all.csv
interval = 96*2
data2 = data.iloc[i:i+interval, :]

df_0 = pd.DataFrame(data2)

max_data = 34739.2

min_data = 11006
# 1441.19

df1 = df_0 * (max_data-min_data) + min_data
df2 = df1.iloc[:,1:]
time_index = list(range(interval))

time = np.array(time_index)
lstn = np.array(df2.iloc[:, 0])
real = np.array(df2.iloc[:, 1])
cnn_lstm = np.array(df2.iloc[:, 2])
cnn_gru = np.array(df2.iloc[:, 3])
gru = np.array(df2.iloc[:, 4])

plt.figure(figsize=(5,5))

x_major_locator = MultipleLocator(96)
y_major_locator = MultipleLocator(2000)
ax = plt.gca()
ax.spines['bottom'].set_linewidth('2.0')
ax.spines['top'].set_linewidth('2.0')
ax.spines['right'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.xaxis.set_major_locator(x_major_locator)#坐标轴分段为8
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel('时间/15minute',fontdict={'size': 16}) #X轴标签
plt.ylabel("电负荷/kWh",fontdict={'size': 16}) #Y轴标签

plt.plot(time, lstn, c='blueviolet',label="LSTN",linewidth=1,linestyle='-')   #
plt.plot(time, cnn_lstm, c='brown', label="CNN-LSTM",linewidth=1,linestyle=':')#
plt.plot(time, real, c='red', label="Real",linewidth=1)
plt.plot(time, cnn_gru, c='yellow', label="CNN-GRU",linewidth=1,linestyle='--')#
plt.plot(time, gru, c='blue', label="GRU",linestyle='--',linewidth=1)#


plt.xlim(0, interval)
plt.ylim(18000, 30500)
plt.legend()
# plt.savefig('C:\\Users\\CaiH\\Desktop\\1.jpg',dpi=800, bbox_inches = 'tight')
# plt.savefig('C:\\Users\\KB9forever\\Desktop\\all-models.png',dpi=800, bbox_inches = 'tight')
# C:\Users\KB9forever\Desktop\毕业设计\LSTNet代码\2.png

plt.show()
# print(df2)