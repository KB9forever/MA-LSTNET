import math
from models import LSTNet
import torch.optim as optim
import pandas as pd
import torch.nn as nn
from utils import *
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR
os.environ["K MP_DUPLICATE_LIB_OK"]  =  "TRUE"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_flag = False
Data = Data_utility(r'C:\Users\CaiH\Desktop\work2-LST\train-valid-dataset\0606数据调整_无时间1.csv', 0.8, 0.18, 12, 7*96)
# Data = Data_utility(r'C:\Users\CaiH\Desktop\work2-LST\train-valid-dataset2\2020-24p-used.c sv', 0.7, 0.2, 12, 7*24)
# one test one file
valid_path = "C:/Users/CaiH/Desktop/LSTM/output/valid.csv"
pred_path = "C:/Users/CaiH/Desktop/LSTM/output/pred.csv"
best_model_load_path = 'C:\\Users\\CaiH\\Desktop\\LSTM\\model\\saved_best_model.pth'
last_model_load_path = 'C:\\Users\\CaiH\\Desktop\\LSTM\\model\\saved_last_model.pth'
# model_load_path = 'C:\\Users\\CaiH\\Desktop\\output\\0608test\\saved_last_model'+str(n)+'.pth'

batch_size = 128
epoch_para =  35
learning_rate = 0.001
best_val  = 10000000
row = 13
step_size =  10
gamma = 1

def evaluate(data, X, Y, model, evaluate1 ,batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):

        X, Y = X.to(device), Y.to(device)
        output = model(X)

        output_np = output.data.cpu().numpy()
        y_np = Y.data.cpu().numpy()

        if predict is None:
            predict = output.detach().cpu().numpy()
            test = Y.detach().cpu().numpy()
        else:
            predict = np.concatenate((predict, output_np), axis=0)
            test = np.concatenate((test, y_np))

        scale = data.scale.expand(output.size(0), data.m)
        scale_min = data.scale_min.expand(output.size(0), data.m)
        ev1 = output * (scale-scale_min) + scale_min
        ev2 = Y * (scale-scale_min) + scale_min

        total_loss += evaluate1(ev1, ev2).item()
        n_samples += (output.size(0) * data.m)


    column1 = ['kw_p']
    column2 = ['kw_r']

    sc = data.scale.cpu().numpy()
    sc_min = data.scale_min.cpu().numpy()
    test_real = test * (sc-sc_min) +  sc_min
    predict_real = predict * (sc-sc_min) +  sc_min
    load_test_real = test_real[:, 12]
    load_predict_real = predict_real[:, 12]
    aa = np.abs((load_test_real - load_predict_real) / load_test_real)
    mape = np.mean(np.abs((load_test_real - load_predict_real) / load_test_real))
    me = np.mean(np.abs(load_test_real - load_predict_real))
    predict_pd = pd.DataFrame(predict[:, row-1], columns=column1)
    test_pd = pd.DataFrame(test[:, row-1], columns=column2)
    result = predict_pd.join(test_pd)
    if test_flag == True:
        result.to_csv(pred_path)
    else:
        result.to_csv(valid_path)

    return math.sqrt(total_loss / n_samples), mape, me


def train(data, X, Y, model, criterion, batch_size, step_size, gamma):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        X, Y = X[:, :, :row].to(device), Y[:, :row].to(device)
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        scale_min = data.scale_min.expand(output.size(0), data.m)

        p1 = output * (scale-scale_min) + scale_min
        p2 = Y * (scale-scale_min) + scale_min

        loss = criterion(p1, p2)#forward[:, 17]
        optimizer.zero_grad()
        loss.backward()#backward
        optimizer.step()#updata
        total_loss += loss.item()#add loss value#负荷预测和实际值算loss
        n_samples += (output.size(0) * data.m)

    scheduler.step()
    return math.sqrt(total_loss / n_samples)

# Set the random seed manually for reproducibility.
torch.manual_seed(54321)
torch.cuda.manual_seed(54321)

# print(Data.rse)
model = LSTNet.Model(Data)
model.to(device)

criterion = nn.MSELoss(size_average=False)
evaluate1 = nn.MSELoss(size_average=False)
criterion = criterion.to(device)
evaluate1 = evaluate1.to(device)

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss_list = []
vai_loss_list = []
# At any point you can hit Ctrl + C to break out of training early.

print('begin training')
for epoch in range(1, epoch_para + 1):

    train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, batch_size,step_size, gamma)
    val_loss, val_mape, val_me = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluate1, batch_size)
    print('| end of epoch {:3d}  | train_loss {:5.4f} | valid_loss {:5.4f} | valid_mape {:5.4f} | valid_me {:5.4f}'.format(
        epoch, train_loss, val_loss, val_mape, val_me))
    # Save the model if the validation loss is the best we've seen so far.

    train_loss_list.append(train_loss)
    vai_loss_list.append(val_loss)

    if val_loss < best_val:
        torch.save(model.state_dict(), best_model_load_path)
        best_val = val_loss
    torch.save(model.state_dict(), last_model_load_path)


#loss visualization
x1 = range(0, epoch_para)
y1 = train_loss_list
y2 = vai_loss_list

plt.xlabel('epoch') #X轴标签
plt.ylabel("loss") #Y轴标签
plt.plot(x1, y1, "-g", label="Train_loss", linewidth=0.6)
plt.plot(x1, y2, "-b", label="Vai_loss", linewidth=0.6)
plt.xlim(0, epoch_para)
plt.legend()
plt.show()

# Load the best saved model.
model = LSTNet.Model(Data)
model.to(device)

state_dict = torch.load(best_model_load_path)
model.load_state_dict(state_dict)
test_flag = True
test_rmse, test_mape, test_me = evaluate(Data, Data.test[0], Data.test[1], model, evaluate1, batch_size)
print("test rmse {:5.4f} | test_mape {:5.4f} | test_me {:5.4f}".format(test_rmse, test_mape, test_me))

# state_dict = torch.load(last_model_load_path)
# model.load_state_dict(state_dict)
# test_rmse = evaluate(Data, Data.test[0], Data.test[1], model, evaluate1, batch_size)
# print("test rmse {:5.4f}  ".format(test_rmse))