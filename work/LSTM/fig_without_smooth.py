import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

i = 1500
data = pd.read_csv('C:\\Users\\CaiH\\Desktop\\work2-LST\\pred-all.csv', header=0)#注意数据的格式，确保都是小数
interval = 96*7
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
lstn_atten = np.array(df2.iloc[:, 5])
lstn_atten_without_cnn = np.array(df2.iloc[:, 6])
plt.figure(figsize=(5,5))

x_major_locator = MultipleLocator(96*7)
y_major_locator = MultipleLocator(2000)
ax = plt.gca()
ax.spines['bottom'].set_linewidth('2.0')
ax.spines['top'].set_linewidth('2.0')
ax.spines['right'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.xaxis.set_major_locator(x_major_locator)#坐标轴分段为8
ax.yaxis.set_major_locator(y_major_locator)
# plt.xlabel('时间/15minute',fontdict={'size': 16}) #X轴标签
# plt.ylabel("电负荷/kWh",fontdict={'size': 16}) #Y轴标签

plt.plot(time, lstn, c='#778899',label="LSTNet",linewidth=1.2,linestyle='-')   #
plt.plot(time, cnn_lstm, c='brown', label="CNN-LSTM",linewidth=1,linestyle=':')#
plt.plot(time, real, c='#DB7093', label="Real",linewidth=1.2)
plt.plot(time, cnn_gru, c='#20B2AA', label="CNN-GRU",linewidth=1,linestyle='--')#
plt.plot(time, gru, c='blue', label="GRU",linestyle='--',linewidth=1)#
plt.plot(time, lstn_atten, c='#00BFFF', label="LSTNet_atten",linestyle='--',linewidth=1.2)#
plt.plot(time, lstn_atten_without_cnn, c='#FF8C00', label="LSTNet-Atten without CNN",linestyle='--',linewidth=1.2)#


plt.xlim(0, interval)
plt.ylim(18000, 32000)
plt.legend()
plt.savefig('C:\\Users\\CaiH\\Desktop\\6.jpg',dpi=800, bbox_inches = 'tight')

plt.show()
# print(df2)
# lstnet_atten900开始
# lstnet1500开始