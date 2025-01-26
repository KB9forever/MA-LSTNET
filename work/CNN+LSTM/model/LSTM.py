import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()
        #self.use_cuda = args.cuda
        self.P = 96 * 7#args.window
        self.m = data.m #列数
        self.num_layers = 1
        self.hidC = 5#args.hidCNN
        self.hidR = 98#args.hidRNN
        self.Ck = 12#args.CNN_kernel
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.LSTM1 = nn.LSTM(self.hidC, self.hidR, self.num_layers, batch_first=False)#hidC input_size,hidden_size
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = torch.sigmoid

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.randn(1, batch_size, self.hidR).to(device)
        c_0 = torch.randn(1, batch_size, self.hidR).to(device)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))#conv1(input )
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.LSTM1(r, (h_0, c_0))
        r1 = self.dropout(torch.squeeze(r[0], 0))

        res = self.linear1(r1)

        if (self.output):
            res = self.output(res)
        return res



