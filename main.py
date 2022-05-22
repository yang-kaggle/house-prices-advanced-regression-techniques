import numpy as np
import matplotlib as plt
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def log_rmse(net, loss, features, labels):
    clipped_pred = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_pred), torch.log(labels)))
    return rmse.item()


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':
    # data preprocess
    train_data = pd.read_csv('./data/house-prices-advanced-regression-techniques/train.csv')
    test_data = pd.read_csv('./data/house-prices-advanced-regression-techniques/test.csv')
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))  # 把第一个特征id删掉
    # 缺失值替换相应特征平均值，特征缩放到零均值和单位方差
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 缺失值填充
    all_features = pd.get_dummies(all_features, dummy_na=True)  # na处理
    # 转换为tensor
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    # net define
    num_inputs = train_features.shape[1]
    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
    )

    # model, parameter, lr, loss define, init
    batch_size, k, num_epochs =64, 5, 100
    lr, weight_decay = 5, 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # k-fold to find best parameter, once done need to commented
    # train_ls_sum, valid_ls_sum = 0, 0
    # for i in range(k):
    #     # prepare train and validation dataset
    #     X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)
    #     house_train = TensorDataset(X_train, y_train)
    #     train_iter = DataLoader(house_train, batch_size=batch_size)
    #     for epoch in range(num_epochs):
    #         for X, y in train_iter:
    #             optimizer.zero_grad()
    #             X, y = X.to(device), y.to(device)
    #             ls = loss(net(X), y)
    #             ls.backward()
    #             optimizer.step()
    #     train_ls = log_rmse(net, loss=loss, features=X_train, labels=y_train)
    #     valid_ls = log_rmse(net, loss=loss, features=X_valid, labels=y_valid)
    #     train_ls_sum += train_ls
    #     valid_ls_sum += valid_ls
    #     print(f'fold {i + 1}, train log rmse {float(train_ls):f}, valid log rmse {float(valid_ls):f}')
    # print(f'{k}-折验证: 平均训练log rmse: {float(train_ls_sum / k):f}, 平均验证log rmse: {float(valid_ls_sum / k):f}')

    # train the model with the parameter from k-fold
    house_train = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(house_train, batch_size=batch_size)
    for epoch in range(num_epochs):
        for X, y in tqdm(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
    train_ls = log_rmse(net, loss=loss, features=house_train[:][0], labels=house_train[:][1])
    print(f'train log rmse {float(train_ls):f}')
    torch.save(net.state_dict(), 'models/linear.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/linear.pth', map_location=device))
    preds = model(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)









