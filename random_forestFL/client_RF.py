import time
import psutil
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils_RF import load_data
import pandas as pd

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.log_data = []  # Lista para armazenar os logs

    def monitor_cpu_energy(self):
        # Função que simula o consumo de energia do CPU (valores fictícios)
        return np.random.uniform(0.5, 2.5)  

    def get_parameters(self, config):
        # Random Forest não usa parâmetros para ajuste como uma rede neural, então retornamos vazio
        return []

    def set_parameters(self, parameters, config):
        # Random Forest não permite ajuste direto de parâmetros, então este método não faz nada
        pass

    def fit(self, parameters, config):
        # Monitoramento do uso de CPU, tempo de treinamento e energia
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent(interval=None)
        cpu_energy_start = self.monitor_cpu_energy()  # Consumo de energia inicial

        # Treinar o modelo usando os dados de treinamento
        X_train, y_train = self.trainloader
        self.model.fit(X_train, y_train)

        # Monitorar final do treinamento
        end_time = time.time()
        cpu_usage_end = psutil.cpu_percent(interval=None)
        training_time = end_time - start_time
        cpu_energy_end = self.monitor_cpu_energy()  # Consumo de energia final
        energy_consumed = cpu_energy_end - cpu_energy_start  # Cálculo fictício de energia consumida

        # Salvar os dados de uso de CPU, tempo de treinamento e energia consumida
        self.log_data.append({
            'Training Time (s)': training_time,
            'CPU Usage (%)': cpu_usage_end,
            'Energy Consumed (W)': energy_consumed  # Energia consumida fictícia
        })

        # Retornar os parâmetros do modelo (vazio) e o número de exemplos
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        # Avaliar o modelo usando os dados de teste
        X_test, y_test = self.testloader
        predictions = self.model.predict(X_test)
        mse_loss = mean_squared_error(y_test, predictions)
        print(f'Evaluation - MSE: {mse_loss:.4f}')

        # Salvar previsões e rótulos para análise futura
        np.savetxt("predictions_client.csv", predictions, delimiter=",")
        np.savetxt("labels_client.csv", y_test, delimiter=",")

        return mse_loss, len(X_test), {"mse": mse_loss}

    def save_logs(self):
        # Salvar os logs em um arquivo CSV
        df_logs = pd.DataFrame(self.log_data)
        df_logs.to_csv('../graphics/graphics_csv/federated_client_cpu_energy_log.csv', index=False)
        print("Logs de CPU, energia e tempo de treinamento salvos em 'federated_client_cpu_energy_log.csv'.")

# Configuração e inicialização do cliente
if __name__ == "__main__":
    # Carregar dados do utilitário (utils_RF.py)
    trainloader, testloader = load_data('../dataset/dataset_FL/datasetDF.csv')

    # Configuração do RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    client = FlowerClient(model, trainloader, testloader)
    
    # Iniciar o cliente federado
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    
    # Salvar logs de CPU e tempo de treinamento
    client.save_logs()
