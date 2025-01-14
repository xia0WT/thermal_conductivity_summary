import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def normalize(data):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    d_max = data.max(dim=0, keepdim=True)[0]
    d_min = data.min(dim=0, keepdim=True)[0]
    return (data - d_min) / (d_max - d_min)
    # return (data - mean) / std

def remove(data, idxes):
    for idx in idxes:
        data = torch.cat((data[:idx, :], data[idx+1:, :]), dim=0)
    return data

def remove_col(data, idxes):
    for idx in idxes:
        data = torch.cat((data[:, :idx], data[:, idx+1:]), dim=1)
    return data


def random_data(data):
    random_indices = torch.randperm(data.size(0))
    shuffled_matrix = data[random_indices]
    return shuffled_matrix


class NNDataset(Dataset):

    def __init__(self, t_data_fp):
        t_data = torch.load(t_data_fp)
        t_data = random_data(t_data)
        label = t_data[:,-1]
        large_label_idxes = reversed(torch.where(label > 100)[0].tolist())
        t_data = remove(t_data, large_label_idxes)
        # t_data = remove(t_data, [155, 140, 137, 134, 132, 125, 114, 113, 99, 97, 91, 88, 86, 85, 81, 67, 64, 60, 52, 47, 42, 41, 40, 19])
        remove_x_idxes = []
        print(t_data.shape)
        for n in range(t_data.shape[1]):
            if t_data[:,n].min() == 0 and t_data[:,n].max() == 0:
                remove_x_idxes.append(n)
        # t_data = remove_col(t_data, reversed(remove_x_idxes))
        print(remove_x_idxes)
        y = t_data[:,-1]
        self.x = t_data[:,:-1]
        scaler = StandardScaler()
        self.x = torch.asarray(scaler.fit_transform(self.x.numpy())).unsqueeze(1)
        # self.x = normalize(t_data[:,:-1]).unsqueeze(1)
        self.y = normalize(t_data[:,-1])
        self.y = torch.sqrt(self.y)
        super(NNDataset, self).__init__()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.x.shape[0]
