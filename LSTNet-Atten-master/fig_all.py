import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

i = 1500
data = pd.read_csv('C:\\Users\\KB9forever\\Desktop\\毕业设计\\LSTNet代码\\pred-all.csv', header=0)#注意数据的格式，确保都是小数
# data = pd.read_csv('C:\\Users\\CaiH\\Desktop\\work2-LST\\pred.csv', header=0)#注意数据的格式，确保都是小数
# C:\\Users\\KB9forever\\Desktop\\毕业设计\\LSTNet代码\\pred-all.csv
interval = 96*1
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

x_major_locator = MultipleLocator(96)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)#坐标轴分段为8
plt.xlabel('时间/15minute') #X轴标签
plt.ylabel("电负荷/kWh") #Y轴标签

x_smooth = np.linspace(np.min(time_index), np.max(time_index),900)#900
model_lstn = make_interp_spline(time, lstn)
model_cnn_lstm = make_interp_spline(time, cnn_lstm)
model_cnn_gru = make_interp_spline(time, cnn_gru)
model_gru = make_interp_spline(time, gru)
model_real = make_interp_spline(time, real)


lstn_smooth = model_lstn(x_smooth)
real_smooth = model_real(x_smooth)
cnn_lstm_smooth = model_cnn_lstm(x_smooth)
cnn_gru_smooth = model_cnn_gru(x_smooth)
gru_smooth = model_gru(x_smooth)


plt.plot(x_smooth, lstn_smooth, c='blueviolet',label="LSTN",linewidth=1.2)   #linestyle='-'
plt.plot(x_smooth, cnn_lstm_smooth, c='brown', label="CNN-LSTM",linewidth=1.2)#,linestyle=':'
plt.plot(x_smooth, real_smooth, c='red', label="Real",linewidth=1.2)
plt.plot(x_smooth, cnn_gru_smooth, c='yellow', label="CNN-GRU",linewidth=1.2)#,linestyle='--'
plt.plot(x_smooth, gru_smooth, c='blue', label="GRU",linewidth=1.2)#linestyle='--',

# plt.figure(figsize=(10,10))
plt.xlim(0, interval)
plt.ylim(15000, 30000)
plt.legend()
plt.savefig('C:\\Users\\KB9forever\\Desktop\\毕业设计\\LSTNet代码\\1.jpg',dpi=800, bbox_inches = 'tight')
# plt.savefig('C:\\Users\\CaiH\\Desktop\\1.jpg',dpi=800, bbox_inches = 'tight')
# C:\\Users\\KB9forever\\Desktop\\毕业设计\\LSTNet代码\\pred-all.csv
plt.show()

# print(df2)