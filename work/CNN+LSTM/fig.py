import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

i = 1500
data = pd.read_csv('C:\\Users\\CaiH\\Desktop\\work2-LST\\CNN+LSTM\\output\\pred.csv', header=0)#注意数据的格式，确保都是小数
interval = 96*10
data2 = data.iloc[i:i+interval, :]

df_0 = pd.DataFrame(data2)

max_data = 34739.2

min_data = 11006
# 1441.19

df1 = df_0 * (max_data-min_data) + min_data
df2 = df1.iloc[:,1:]
time_index = list(range(interval))

time = np.array(time_index)
pred = np.array(df2.iloc[:, 0])
real = np.array(df2.iloc[:, 1])

x_major_locator = MultipleLocator(96)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)#坐标轴分段为8
plt.xlabel('time(15 minute interval)') #X轴标签
plt.ylabel("kWh") #Y轴标签

x_smooth = np.linspace(np.min(time_index), np.max(time_index),900)
model_pred = make_interp_spline(time, pred)
model_real = make_interp_spline(time, real)
pred_smooth = model_pred(x_smooth)
real_smooth = model_real(x_smooth)

plt.plot(x_smooth, pred_smooth, c='green', marker='*', ms=1, alpha=0.75, label="prediction")
plt.plot(x_smooth, real_smooth, c='red', marker='o', ms=1, alpha=0.75, label="real")
plt.xlim(0, interval)
# plt.ylim(12000, 28000)
plt.legend()
plt.show()
# print(df2)