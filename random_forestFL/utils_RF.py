from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

# Definir a classe do Dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.iloc[:, :-2] = (self.data.iloc[:, :-2] - self.data.iloc[:, :-2].mean()) / self.data.iloc[:, :-2].std()  # Normalizar características

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-2].values.astype(np.float32)  # Excluir as duas últimas colunas
        labels = self.data.iloc[idx, -2:].values.astype(np.float32)  # As duas últimas colunas são os rótulos
        return features, labels  # Retornar numpy arrays

def load_data(csv_file):
    dataset = CustomDataset(csv_file)
    train_size = int(0.7 * len(dataset))  # 70% dos dados para treinamento
    test_size = len(dataset) - train_size  # 30% dos dados para teste
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    X_train = np.array([data[0] for data in trainset])
    y_train = np.array([data[1] for data in trainset])
    X_test = np.array([data[0] for data in testset])
    y_test = np.array([data[1] for data in testset])

    return (X_train, y_train), (X_test, y_test)

# Função de treinamento para Random Forest
def train(model, trainloader, criterion=None, optimizer=None, device=None):
    X_train, y_train = trainloader
    model.fit(X_train, y_train)

# Função de teste para Random Forest
def test(model, testloader, criterion=None, device=None):
    X_test, y_test = testloader
    predictions = model.predict(X_test)
    mse_loss = mean_squared_error(y_test, predictions)
    return mse_loss
