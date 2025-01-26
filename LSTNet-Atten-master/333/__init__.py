import torch
import math
import torch.nn as nn
from models import LSTNet
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Data = Data_utility(r'C:\Users\CaiH\Desktop\work2-LST\train-valid-dataset\0606数据调整_无时间.csv', 0.6, 0.2, 12, 7*96)
last_model_load_path = 'C:\\Users\\CaiH\\Desktop\\output\\0608test\\saved_last_model.pth'
best_model_load_path = 'C:\\Users\\CaiH\\Desktop\\output\\0608test\\saved_best_model.pth'

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

    # column1 = ['kw_p']
    # column2 = ['kw_r']
    # predict_pd = pd.DataFrame(predict[:, row-1], columns=column1)
    # test_pd = pd.DataFrame(test[:, row-1], columns=column2)
    # result = predict_pd.join(test_pd)
    # if test_flag == True:
    #     result.to_csv(pred_path)
    # else:
    #     result.to_csv(valid_path)

    return math.sqrt(total_loss / n_samples)

evaluate1 = nn.MSELoss(size_average=False)
evaluate1 = evaluate1.to(device)

model = LSTNet.Model(Data)
model.to(device)
state_dict = torch.load(best_model_load_path)
model.load_state_dict(state_dict)

test_rmse = evaluate(Data, Data.test[0], Data.test[1], model, evaluate1, 128)
print("test rmse {:5.4f}  ".format(test_rmse))