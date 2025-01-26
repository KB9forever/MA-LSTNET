import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()
        #self.use_cuda = args.cuda
        self.P = 96 * 7#args.window
        self.m = data.m #列数
        self.hidC = 5#args.hidCNN
        self.hidR = 36#args.hidRNN

        self.hidS = 10#args.hidSkip
        self.Ck = 6#args.CNN_kernel
        self.skip = 96#args.skip
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = 96 #args.highway_window96
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.m, self.hidR)#hidC input_size,hidden_size
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = torch.sigmoid
        self.multihead = nn.MultiheadAttention(embed_dim=self.hidR, num_heads=2)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.m, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS + self.hidR, self.m)
            # self.linear1 = nn.Linear(self.hidR + self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)


    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        # c = x.view(-1, 1, self.P, self.m)
        # c = F.relu(self.conv1(c))#conv1(input )
        # c = self.dropout(c)
        # c = torch.squeeze(c, 3)

        # RNN
        r = x.permute(1, 0, 2).contiguous()
        H_t, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        c = x.permute(0, 2, 1).contiguous()
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.m, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.m)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)

            r = torch.cat((r, s), 1)

        # multi-head-attention
        a, _ = self.multihead(H_t, H_t, H_t)
        a = a.permute(1, 0, 2)[:, -1, :]
        r = torch.cat((a, r), 1)
        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res



