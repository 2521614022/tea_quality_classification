from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

def r2_score(y_true, y_pred):
        y_true_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        if ss_tot == 0:
            return 0
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
def log_rmse(y_true, y_pred):
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)
        loss = nn.MSELoss()
        rmse = torch.sqrt(loss(torch.log(y_pred),torch.log(y_true)))
        return rmse.item()

def random_forest_regression():
    train_data = pd.read_csv("./data/yd_features_norm_train.csv")
    test_data = pd.read_csv("./data/yd_features_norm_test.csv")
    all_features = pd.concat((train_data.iloc[:,:-1], test_data.iloc[:,:-1]))
    all_labels = pd.concat((train_data.iloc[:,-1], test_data.iloc[:,-1]))
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values,
                                  dtype=torch.float32).numpy()
    test_features = torch.tensor(all_features[n_train:].values,
                                 dtype = torch.float32).numpy()
    train_labels = torch.tensor(all_labels[:n_train].values.reshape(-1,),
                                dtype = torch.float32).numpy()
    test_labels = torch.tensor(all_labels[n_train:].values.reshape(-1,),
                                dtype = torch.float32).numpy()
     
    # pca = PCA(n_components=3)
    # train_features = torch.tensor(pca.fit_transform(train_features), dtype = torch.float32).numpy()
    # test_features = torch.tensor(pca.fit_transform(test_features), dtype = torch.float32).numpy()
    
    # Creating a random forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=43)
    rf_regressor.fit(train_features, train_labels)
    y_pred = rf_regressor.predict(test_features)
    print(y_pred)
    print(f'r2_score:{r2_score(test_labels, y_pred):.8f}')
    print(f'log_rmse:{log_rmse(test_labels, y_pred):.8f}')
    
def gradient_boosting_regression():
    train_data = pd.read_csv("./data/yd_features_norm_train.csv")
    test_data = pd.read_csv("./data/yd_features_norm_test.csv")
    all_features = pd.concat((train_data.iloc[:,:-1], test_data.iloc[:,:-1]))
    all_labels = pd.concat((train_data.iloc[:,-1], test_data.iloc[:,-1]))
    n_train = train_data.shape[0] 
    train_features = torch.tensor(all_features[:n_train].values,
                                  dtype=torch.float32).numpy()
    test_features = torch.tensor(all_features[n_train:].values,
                                 dtype = torch.float32).numpy()
    train_labels = torch.tensor(all_labels[:n_train].values.reshape(-1,),
                                dtype = torch.float32).numpy()
    test_labels = torch.tensor(all_labels[n_train:].values.reshape(-1,),
                                dtype = torch.float32).numpy()
      
    # pca = PCA(n_components=3) 
    # train_features = torch.tensor(pca.fit_transform(train_features), dtype = torch.float32).numpy()
    # test_features = torch.tensor(pca.fit_transform(test_features), dtype = torch.float32).numpy()
    
    # Create a gradient lifting regression tree model
    gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=43)
    gbrt.fit(train_features, train_labels)
    y_pred = gbrt.predict(test_features)
    print(y_pred)
    print(f'r2_score:{r2_score(test_labels, y_pred):.8f}')
    print(f'log_rmse:{log_rmse(test_labels, y_pred):.8f}')
    
def support_vector_regression():
    train_data = pd.read_csv("./data/yd_features_norm_train.csv")
    test_data = pd.read_csv("./data/yd_features_norm_test.csv")
    all_features = pd.concat((train_data.iloc[:,:-1], test_data.iloc[:,:-1]))
    all_labels = pd.concat((train_data.iloc[:,-1], test_data.iloc[:,-1]))
    n_train = train_data.shape[0] 
    train_features = torch.tensor(all_features[:n_train].values,
                                  dtype=torch.float32).numpy()
    test_features = torch.tensor(all_features[n_train:].values,
                                 dtype = torch.float32).numpy()
    train_labels = torch.tensor(all_labels[:n_train].values.reshape(-1,),
                                dtype = torch.float32).numpy()
    test_labels = torch.tensor(all_labels[n_train:].values.reshape(-1,),
                                dtype = torch.float32).numpy()
      
    # pca = PCA(n_components=3)  
    # train_features = torch.tensor(pca.fit_transform(train_features), dtype = torch.float32).numpy()
    # test_features = torch.tensor(pca.fit_transform(test_features), dtype = torch.float32).numpy()
    
    # Creating a support vector regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.5)
    svr_rbf.fit(train_features, train_labels)
    y_pred = svr_rbf.predict(test_features)
    print(y_pred)
    print(f'r2_score:{r2_score(test_labels, y_pred):.8f}')
    print(f'log_rmse:{log_rmse(test_labels, y_pred):.8f}')
    
def xgboost_regression():
    train_data = pd.read_csv("./data/yd_features_norm_train.csv")
    test_data = pd.read_csv("./data/yd_features_norm_test.csv")
    all_features = pd.concat((train_data.iloc[:,:-1], test_data.iloc[:,:-1]))
    all_labels = pd.concat((train_data.iloc[:,-1], test_data.iloc[:,-1]))
    n_train = train_data.shape[0] 
    train_features = torch.tensor(all_features[:n_train].values,
                                  dtype=torch.float32).numpy()
    test_features = torch.tensor(all_features[n_train:].values,
                                 dtype = torch.float32).numpy()
    train_labels = torch.tensor(all_labels[:n_train].values.reshape(-1,),
                                dtype = torch.float32).numpy()
    test_labels = torch.tensor(all_labels[n_train:].values.reshape(-1,),
                                dtype = torch.float32).numpy()
     
    # pca = PCA(n_components=3)
    # train_features = torch.tensor(pca.fit_transform(train_features), dtype = torch.float32).numpy()
    # test_features = torch.tensor(pca.fit_transform(test_features), dtype = torch.float32).numpy()
    
    dtrain = xgb.DMatrix(train_features, train_labels)
    dtest = xgb.DMatrix(test_features, test_labels)
    # Set XGBoost parameters
    param = {
        'max_depth': 6, 
        'eta': 0.05, 
        'objective': 'reg:squarederror', 
    }
    num_round = 200
    bst = xgb.train(param, dtrain, num_round)
    y_pred = bst.predict(dtest)
    print(y_pred)
    print(f'r2_score:{r2_score(test_labels, y_pred):.8f}')
    print(f'log_rmse:{log_rmse(test_labels, y_pred):.8f}')
    
if __name__ == '__main__':
    # random_forest_regression()
    # gradient_boosting_regression()
    # support_vector_regression()
    xgboost_regression()