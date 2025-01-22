import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(10, 6)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('#ffd323', '#88aa29', '#ef3054'), figsize=(10, 6), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt, linewidth=4)
        else:
            axes.plot(y, fmt, linewidth=4)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def main():
    train_data = pd.read_csv("../data/sg_features_norm_train.csv")
    test_data = pd.read_csv("../data/sg_features_norm_test.csv")
    all_features = pd.concat((train_data.iloc[:,:], test_data.iloc[:,:]))

    # Extract the Numpy format from the pandas format and convert it to a tensor representation
    n_train = train_data.shape[0] # Number of samples
    train_features = torch.tensor(all_features[:n_train].values,
                                    dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values,
                                    dtype=torch.float32)
    train_labels = torch.tensor(train_data.scores.values.reshape(-1,1),
                                dtype = torch.float32)
    test_labels = torch.tensor(test_data.scores.values.reshape(-1,1),
                                dtype = torch.float32)

    # Training
    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    def get_net():
            # net = nn.Sequential(nn.Linear(in_features,1)) 
            net = nn.Sequential(nn.Linear(in_features,256),nn.ReLU(),
                                nn.Linear(256,128),nn.ReLU(),
                                nn.Linear(128,64),nn.ReLU(),
                                nn.Linear(64,1)) # Multilayer perceptron
            return net
        
    def log_rmse(net,features,labels):
            clipped_preds = torch.clamp(net(features),1,float('inf')) # Limit the output of the model between 1 and inf
            rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
            return rmse.item()
        
    # Training function with the help of Adam optimizer
    def train(net, train_features, train_labels, test_features,test_labels,
                num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [],[]
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X),y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net,train_features,train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        # Save the model
        torch.save(net.state_dict(), '../models/MLP-SG-4.pt')
        return train_ls, test_ls

    # K-fold cross-validation
    def get_k_fold_data(k,i,X,y):
        assert k > 1
        fold_size = X.shape[0] // k # The size of each fold is number of samples divided by k
        X_train, y_train = None,None
        for j in range(k):
            idx = slice(j*fold_size,(j+1)*fold_size) # Slice index for each fold
            X_part, y_part = X[idx,:], y[idx] 
            if j == i: 
                X_valid, y_valid = X_part, y_part
            elif X_train is None: 
                X_train, y_train = X_part, y_part
            else: 
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid 

    def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train) 
            net = get_net()
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size) 
            
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            
            if i == 0:
                plot(list(range(1,num_epochs+1)),[train_ls,valid_ls],
                            xlabel='epoch',ylabel='rmse',xlim=[1,num_epochs],
                            legend=["train","valid"],yscale='log')
            print(f'fold{i+1},train log rmse {float(train_ls[-1]):f},'
                    f'valid log rmse {float(valid_ls[-1]):f}')
        return train_l_sum / k, valid_l_sum / k 

    k = 5
    num_epochs = 200
    lr = 1e-5
    weight_decay = 0
    batch_size = 128
    # plt.figure(figsize=(10,6))
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # print(f'{k}-Fold verification：Average training log rmse：{float(train_l):f},'f'Average validation log rmse：{float(valid_l):f}')
    # plt.grid(visible=False)
    # plt.show()

    def r2_score(y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
        
    def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
        net = get_net()
        train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
        plot(np.arange(1, num_epochs+1),[train_ls], xlabel='epoch',
                    ylabel = 'rmse', xlim=[1,num_epochs], yscale='log')
        print(f'train log rmse {float(train_ls[-1]):f}')
        preds = net(test_features).detach().numpy()
        # test_data['pred_scores'] = pd.Series(preds.reshape(1,-1)[0])
        # submission = pd.concat([test_data['scores'],test_data['pred_scores']],axis=1)
        # submission.to_csv('../results/sg_submission.csv',index = False)
        print(f'r2_score:{r2_score(test_labels, preds):.8f}')
        print(f'log_rmse:{log_rmse(net, test_features, test_labels):.8f}')
        
    plt.figure(figsize=(10,6))
    train_and_pred(train_features, test_features, train_labels, test_data,
                num_epochs, lr, weight_decay, batch_size)
    plt.grid(visible=False)
    plt.show()
    
if __name__ == "__main__":
    main()