import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#图数据构建
def create_data(residual,mass,motion,label):

    x = torch.cat((residual[:3652],motion,mass),dim=1).float()
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    y =label.float()
    data_list = []
    for i in range(len(residual)):
        data = Data(x=x[i].unsqueeze(1), edge_index=edge_index, y=y[i])
        data_list.append(data)
    return data_list

#数据集加载
residual = np.load('E:\\Workplace\\P\\2012-2022\\residual\\data\\residual2012_2021.npy')
motion = np.load('E:\\Workplace\\P\\2012-2022\\residual\\data\\motion_2012_daily.npy')[:3652]
mass = np.load('E:\\Workplace\\P\\2012-2022\\residual\\data\\mass_2012_daily.npy')[:3652]

train_label = torch.from_numpy(residual)[13:3643].unsqueeze(1)
train_residual = torch.from_numpy(residual)[3:3633].unsqueeze(1)
train_motion = torch.from_numpy(motion)[3:3633].unsqueeze(1)
train_mass = torch.from_numpy(mass)[3:3633].unsqueeze(1)

train_data = create_data(train_residual,train_mass,train_motion,train_label)
train_data_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_mass = torch.from_numpy(mass)[3633:3643].unsqueeze(1)
test_residual = torch.from_numpy(residual)[3633:3643].unsqueeze(1)
test_motion = torch.from_numpy(motion)[3633:3643].unsqueeze(1)
test_data = create_data(test_residual,test_mass,test_motion,test_residual)
target_data = residual[-10:]
test_data_loader = DataLoader(test_data,batch_size=10,shuffle=False)

#图注意力神经网络
class SimpleGNNWithAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(SimpleGNNWithAttention, self).__init__()
        self.conv1 = GATConv(1, hidden_dim, heads=1, concat=True, negative_slope=0.2)  # 输入维度为1，输出维度为16
        self.conv2 = GATConv(hidden_dim, 1, heads=1, concat=True, negative_slope=0.2)  # 输入维度为16，输出维度为1
        self.linear = nn.Linear(30, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x.view(-1, 30))

        return x
#图LSTM
class SimpleGNNWithAttentionLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleGNNWithAttentionLSTM, self).__init__()
        self.conv1 = GATConv(1, hidden_dim, heads=1, concat=True, negative_slope=0.2)
        self.conv2 = GATConv(hidden_dim, 1, heads=1, concat=True, negative_slope=0.2)
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.linear = nn.Linear(30, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = x.unsqueeze(0)
        # Assuming x has the shape (batch_size, num_nodes, hidden_dim)
        # You may need to adjust the dimensions depending on your data
        x, _ = self.lstm(x)

        # Take the last hidden state of the LSTM
        x = x.squeeze(0)

        x = self.linear(x.view(-1, 30))

        return x

#贝叶斯优化的目标函数，以验证集结果为输出
def target_function(epochs,lr,hidden_num):
    #模型训练及验证
    epochs = int(epochs)
    hidden_num = int(hidden_num)
    model = SimpleGNNWithAttentionLSTM(hidden_dim=hidden_num)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.cuda()
    criterion = criterion.cuda()
    #训练
    model.train()
    for epoch in range(epochs):
        for batch in train_data_loader:
            optimizer.zero_grad()
            batch = batch.cuda()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
    #验证
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            batch = batch.cuda()
            pred = model(batch)
            pred = pred.squeeze().cpu().detach().numpy()
    MAE = np.abs(np.sum(pred - target_data) / len(pred))
    plt.clf()
    plt.plot(pred, label='pred')
    plt.plot(target_data, label='observed')
    plt.legend()
    plt.savefig('E:\\Workplace\\P\\2012-2022\\residual\\fig\\epoch{}hn{}lr{}.png'.format(epochs,hidden_num,lr))
    # plt.show()
    print(pred)
    return -MAE

#参数集
pbounds = {
    'epochs':(1,50),
    'hidden_num':(5,50),
    'lr':(0.001,0.1),
}
#贝叶斯优化
optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=42  # 随机种子，用于确保结果的可重复性
)
optimizer.maximize(init_points=10, n_iter=100)
