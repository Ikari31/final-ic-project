import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data, LinearRegression, train, test  # Importar do arquivo auxiliar

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.MSELoss()  # Usar MSELoss para regressão
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters, config):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.model, self.trainloader, self.criterion, self.optimizer, self.device)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        mse_loss, predictions, labels = test(self.model, self.testloader, self.criterion, self.device)
        print(f'Evaluation - MSE: {mse_loss:.3f}')
        
        # Salvar previsões e rótulos em arquivos CSV
        np.savetxt("predictions_client.csv", predictions, delimiter=",")
        np.savetxt("labels_client.csv", labels, delimiter=",")
        
        return mse_loss, len(self.testloader.dataset), {"mse": mse_loss}


# Configuração e inicialização do cliente
if __name__ == "__main__":
    trainloader, testloader = load_data('../dataset/dataset_FL/datasetDF.csv')
    input_size = trainloader.dataset[0][0].shape[0]
    output_size = 2

    model = LinearRegression(input_size, output_size)
    client = FlowerClient(model, trainloader, testloader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)