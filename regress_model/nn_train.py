import torch
from torch import nn
from torch.utils.data import random_split
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score

# from regress_model.nn_model import NNModel
from regress_model.cnn_model import DOSCNN as NNModel
from regress_model.nn_dataset import NNDataset


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Train:

    def __init__(self):
        dataset = NNDataset("../data/total_dos_0-76hz_4096d.pt")
        # dataset = NNDataset("../data/total_dos_2-22hz_4096d.pt")
        split_data = random_split(dataset, [0.6, 0.4])
        self.train_loader = DataLoader(split_data[0], batch_size=5, shuffle=True)
        self.test_loader = DataLoader(split_data[1], batch_size=5, shuffle=True)
        # self.model = NNModel(3118)
        # self.model = NNModel(4096)
        self.model = NNModel(4096)
        # self.model.load_state_dict(torch.load('./models.pt'))
        self.model.apply(init_weights)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5, )

    def train_epoches(self):
        with tqdm(total=2000) as pbar:
            for e in range(2000):
                train_loss, train_real, train_pred = self.train()
                test_loss, test_real, test_pred = self.test(False)
                if e % 100 == 0:
                    self.draw(train_real, train_pred, test_real, test_pred)
                    torch.save(self.model.state_dict(), f"./models.pt")

                pbar.set_postfix_str(f"train_loss: {train_loss}, test_loss: {test_loss}")
                pbar.update(1)
                l = self.scheduler.get_last_lr()[0]
                if self.scheduler.get_last_lr()[0] > 0.0001:
                    self.scheduler.step()

    def train(self):
        self.model.train()
        sum_loss = 0
        real = []
        pred = []
        # 下面使用了train_data_loader，循环中每次会获得batch_size个数据
        for batch_features, batch_real_prices in self.train_loader:
            batch_pred_prices = self.model(batch_features)
            loss = self.loss_func(batch_pred_prices, batch_real_prices)
            sum_loss += loss.item() * len(batch_features)
            self.optimizer.zero_grad()  # 将优化器的梯度归零
            loss.backward()  # 反向传播，计算梯度d loss/d x
            self.optimizer.step()  # 更新所有参数
            real.extend(batch_real_prices.cpu().tolist())
            pred.extend(batch_pred_prices.view(-1).cpu().tolist())
        return sum_loss / len(self.train_loader), real, pred

    def test(self, show_fig):
        self.model.eval()
        sum_loss = 0
        real = []
        pred = []
        for batch_features, batch_real_prices in self.test_loader:
            batch_pred_prices = self.model(batch_features)
            loss = self.loss_func(batch_pred_prices, batch_real_prices)
            sum_loss += loss.item() * len(batch_features)
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            real.extend(batch_real_prices.cpu().tolist())
            pred.extend(batch_pred_prices.view(-1).cpu().tolist())
        if show_fig:
            r2 = r2_score(real, pred)
            plt.scatter(real, pred)
            plt.plot([0, 1], [0, 1])
            plt.xlabel("real")
            plt.ylabel("pred")
            plt.text(0.8, 0.8, f"r2 = {r2}")
            plt.show()
        return sum_loss / len(self.test_loader), real, pred

    def draw(self, train_real, train_pred, test_real, test_pred):
        plt.scatter(train_real, train_pred, label="train")
        plt.scatter(test_real, test_pred, label="test")
        plt.plot([0, 1], [0, 1])
        plt.xlabel("real")
        plt.ylabel("pred")
        plt.text(0.8, 0.8, f"train r2 = {round(r2_score(train_real, train_pred), 2)}")
        plt.text(0.8, 0.6, f"test r2 = {round(r2_score(test_real, test_pred), 2)}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    Train().train_epoches()
