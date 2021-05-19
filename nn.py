import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as pjoin
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import DataSet
import optuna


class Net(nn.Module):
    def __init__(self, c1=32, c2=64, c3=128, c4=256, l1=512, d1=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(9, c1, (5, 5))
        self.conv2 = nn.Conv2d(c1, c2, (5, 5))
        self.conv3 = nn.Conv2d(c2, c3, (5, 5))
        self.conv4 = nn.Conv2d(c3, c4, (5, 5))
        x = torch.randn(900, 900).view(-1, 9, 100, 100)
        self.toLinear = -1
        self.convs(x)
        self.fc1 = nn.Linear(self.toLinear, l1)
        self.fc2 = nn.Linear(l1, 2)
        self.dropout1 = nn.Dropout(d1)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        if self.toLinear == -1:
            self.toLinear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.toLinear)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


LR = 0.0006740374815727881
writer = SummaryWriter()
net = Net()
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_fun = nn.MSELoss()
epochloss = 1.0
crossloss = 1.0
bestcross = 0
PATIENCE = 10
patience = PATIENCE
TYPE = "LVOT"
OPT = False


tf = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(90)
])


def objective(trial):
    global bestcross
    global crossloss
    global net
    global optimizer
    global epochloss
    c1 = trial.suggest_int(name="c1", low=32, high=64, step=32)
    c2 = trial.suggest_int(name="c2", low=64, high=128, step=32)
    c3 = trial.suggest_int(name="c3", low=128, high=256, step=32)
    c4 = trial.suggest_int(name="c4", low=256, high=512, step=32)
    l1 = trial.suggest_int(name="l1", low=128, high=1024, step=32)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    d1 = trial.suggest_float(name="d1", low=0.0, high=0.6, step=0.1)
    del net
    net = Net(c1, c2, c3, c4, l1, d1)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    dataset = DataSet(TYPE, "/training_data.npy", transform=None)
    _X = []
    _Y = []
    for data in dataset:
        _X.append(data[0])
        _Y.append(data[1])
    X = torch.tensor(_X)
    Y = torch.tensor(_Y)
    EPOCHS = 100
    BATCH_SIZE = 600
    K = 10
    for cross in range(K):
        start = round(len(X) * 0.1) * cross
        stop = round(len(X) * 0.1) * (cross + 1)
        print("CROSS: ", cross + 1)
        if cross == 0:
            train_x = X[stop:]
            train_y = Y[stop:]
        elif cross == 9:
            train_x = X[:start]
            train_y = Y[:start]
        else:
            train_x = torch.cat((X[0:start], X[stop:]), 0)
            train_y = torch.cat((Y[0:start], Y[stop:]), 0)
        valid_x = X[start:stop]
        valid_y = Y[start:stop]
        for epoch in range(EPOCHS):
            print(epoch + 1)
            for batch in range(0, len(train_x), BATCH_SIZE):
                batch_x = train_x[batch:batch + BATCH_SIZE]
                batch_y = train_y[batch:batch + BATCH_SIZE]
                acc, loss = fwd_pass(batch_x.view(-1, 9, 100, 100).float(), batch_y.float(), True)
                print(round(acc, 3), loss)
                writer.add_scalar("Training loss", loss, epoch)
            valid_result, valid_loss = valid(epoch, valid_x, valid_y, cross)
            if not valid_result:
                break
        if epochloss < crossloss:
            crossloss = epochloss
            bestcross = cross
        epochloss = 1.0
        del net
        net = Net(c1, c2, c3, c4, l1, d1)
        optimizer = optim.Adam(net.parameters(), lr=lr)
    net = torch.load(pjoin(TYPE, str(bestcross + 1)))
    net.eval()
    acc = test()
    return acc


def resetmodel():
    global net
    global optimizer
    global epochloss
    epochloss = 1.0
    del net
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=LR)


def train():
    resetmodel()
    dataset = DataSet(TYPE, "/training_data.npy", transform=None)
    _X = []
    _Y = []
    for data in dataset:
        _X.append(data[0])
        _Y.append(data[1])
    X = torch.tensor(_X)
    Y = torch.tensor(_Y)
    EPOCHS = 100
    BATCH_SIZE = 600
    K = 10
    for cross in range(K):
        start = round(len(X) * 0.1) * cross
        stop = round(len(X) * 0.1) * (cross + 1)
        print("CROSS: ", cross + 1)
        if cross == 0:
            train_x = X[stop:]
            train_y = Y[stop:]
        elif cross == 9:
            train_x = X[:start]
            train_y = Y[:start]
        else:
            train_x = torch.cat((X[0:start], X[stop:]), 0)
            train_y = torch.cat((Y[0:start], Y[stop:]), 0)
        valid_x = X[start:stop]
        valid_y = Y[start:stop]
        for epoch in range(EPOCHS):
            print(epoch + 1)
            for batch in range(0, len(train_x), BATCH_SIZE):
                batch_x = train_x[batch:batch+BATCH_SIZE]
                batch_y = train_y[batch:batch+BATCH_SIZE]
                acc, loss = fwd_pass(batch_x.view(-1, 9, 100, 100).float(), batch_y.float(), True)
                print(round(acc, 3), loss)
                writer.add_scalar("Training loss", loss, epoch)
            valid_result, valid_loss = valid(epoch, valid_x, valid_y, cross)
            if not valid_result:
                break
        global crossloss
        global bestcross
        if epochloss < crossloss:
            crossloss = epochloss
            bestcross = cross
        resetmodel()


def valid(e, x, y, c):
    with torch.no_grad():
        acc, loss = fwd_pass(x.view(-1, 9, 100, 100).float(), y.float(), False)
        print("VALID: ", round(acc, 3), loss)
        writer.add_scalar("Valid loss", loss, e)
        global epochloss
        global patience
        if loss < epochloss:
            epochloss = loss
            patience = PATIENCE
            torch.save(net, pjoin(TYPE, str(c + 1)))
        else:
            patience -= 1
        if patience != 0:
            return True, loss
        else:
            return False, loss


def test():
    dataset = DataSet(TYPE, "/test_data.npy", transform=None)
    _X = []
    _Y = []
    for data in dataset:
        _X.append(data[0])
        _Y.append(data[1])
    X = torch.tensor(_X)
    Y = torch.tensor(_Y)
    # writer.add_graph(net, X[0].view(-1, 9, 100, 100).float())
    with torch.no_grad():
        acc, loss = fwd_pass(X.view(-1, 9, 100, 100).float(), Y.float(), False)
        print("TEST: ", round(acc, 3), loss)
        return acc


def fwd_pass(x, y, trainf):
    if trainf:
        optimizer.zero_grad()
    outputs = net(x)
    loss = loss_fun(outputs, y)
    if trainf:
        loss.backward()
        optimizer.step()
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    return acc, loss


if __name__ == '__main__':
    if OPT:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        train()
        net = torch.load(pjoin(TYPE, str(bestcross + 1)))
        net.eval()
        test()
        torch.save(net, pjoin(TYPE, "Model"))
    writer.close()
    print("STOP")
