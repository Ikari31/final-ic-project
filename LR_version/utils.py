import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
        return torch.tensor(features), torch.tensor(labels)

def load_data(csv_file):
    dataset = CustomDataset(csv_file)
    train_size = int(0.7 * len(dataset))  # 70% dos dados para treinamento
    test_size = len(dataset) - train_size  # 30% dos dados para teste
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader

# Definir o modelo de regressão linear
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def test(model, testloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels).item()
            total_loss += loss * labels.size(0)  # Acumular a perda ponderada pelo número de amostras no batch
            total_samples += labels.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples  # Média ponderada da perda
    return avg_loss, np.vstack(all_predictions), np.vstack(all_labels)
