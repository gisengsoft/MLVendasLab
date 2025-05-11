from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import joblib
import os

class ModeloVendasSorvete:
    def __init__(self):
        self.modelo = LinearRegression()
        self.metricas = {}
    
    def treinar(self, X_train, y_train):
        """Treina o modelo com os dados fornecidos."""
        self.modelo.fit(X_train, y_train)
        print("Modelo treinado com sucesso!")
        print(f"Coeficientes: {self.modelo.coef_}")
        print(f"Intercepto: {self.modelo.intercept_}")
        return self.modelo
    
    def avaliar(self, X_test, y_test):
        """Avalia o modelo com métricas de regressão."""
        y_pred = self.modelo.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        self.metricas = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        print("Métricas de avaliação:")
        for metric_name, metric_value in self.metricas.items():
            print(f"{metric_name}: {metric_value:.2f}")
        
        return self.metricas
    
    def visualizar_resultados(self, X, y, X_test=None, y_test=None, y_pred=None):
        """Cria visualizações para entender os resultados do modelo."""
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Dados e linha de regressão
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X.flatten(), y=y, color='blue', alpha=0.6)
        
        # Criar linha de tendência
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred_line = self.modelo.predict(x_range)
        
        plt.plot(x_range, y_pred_line, color='red', linewidth=2)
        plt.title('Vendas x Temperatura')
        plt.xlabel('Temperatura (°C)')
        plt.ylabel('Vendas de Sorvete')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Previsão vs Real (se disponível)
        if X_test is not None and y_test is not None and y_pred is not None:
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, y_pred, color='green', alpha=0.6)
            
            # Linha identidade (y=x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Valores Reais vs. Previstos')
            plt.xlabel('Vendas Reais')
            plt.ylabel('Vendas Previstas')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar figura
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/resultados_modelo.png')
        
        plt.show()
        return plt
    
    def prever(self, temperatura):
        """Faz previsões para novas temperaturas."""
        if isinstance(temperatura, (int, float)):
            temperatura = np.array([[temperatura]])
        elif isinstance(temperatura, list):
            temperatura = np.array(temperatura).reshape(-1, 1)
            
        previsoes = self.modelo.predict(temperatura)
        return previsoes
    
    def salvar_modelo(self, caminho='outputs/modelo_vendas_sorvete.joblib'):
        """Salva o modelo treinado."""
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        joblib.dump(self.modelo, caminho)
        print(f"Modelo salvo em: {caminho}")
    
    def carregar_modelo(self, caminho='outputs/modelo_vendas_sorvete.joblib'):
        """Carrega um modelo salvo."""
        self.modelo = joblib.load(caminho)
        print(f"Modelo carregado de: {caminho}")
        return self.modelo
    
    def registrar_modelo_mlflow(self, X_train, y_train, X_test, y_test, 
                               run_name="VendasSorvete_Regressao"):
        """Registra o modelo e métricas usando MLflow."""
        mlflow.set_experiment("Previsao_Vendas_Sorvete")
        
        with mlflow.start_run(run_name=run_name):
            # Treinar modelo
            self.treinar(X_train, y_train)
            
            # Avaliar modelo
            y_pred = self.modelo.predict(X_test)
            metricas = self.avaliar(X_test, y_test)
            
            # Registrar parâmetros
            mlflow.log_param("model_type", "LinearRegression")
            
            # Registrar métricas
            for metric_name, metric_value in metricas.items():
                mlflow.log_metric(metric_name.lower(), metric_value)
            
            # Criar e salvar visualizações
            fig = self.visualizar_resultados(
                np.vstack((X_train, X_test)), 
                np.concatenate((y_train, y_test)),
                X_test, y_test, y_pred
            )
            
            # Salvar figura para MLflow
            fig_path = 'outputs/temp_plot.png'
            fig.savefig(fig_path)
            mlflow.log_artifact(fig_path, "plots")
            
            # Registrar modelo
            mlflow.sklearn.log_model(self.modelo, "model")
            
            print(f"Modelo registrado no MLflow com run_id: {mlflow.active_run().info.run_id}")
            
            return mlflow.active_run().info.run_id

if __name__ == "__main__":
    # Testar a classe
    from pre_processamento import carregar_dados, preparar_dados
    
    dados = carregar_dados("inputs/base_vendas_sorvete.csv")
    X_train, X_test, y_train, y_test = preparar_dados(dados)
    
    modelo = ModeloVendasSorvete()
    modelo.treinar(X_train, y_train)
    
    y_pred = modelo.prever(X_test)
    metricas = modelo.avaliar(X_test, y_test)
    
    modelo.visualizar_resultados(
        np.vstack((X_train, X_test)), 
        np.concatenate((y_train, y_test)),
        X_test, y_test, y_pred
    )
    
    modelo.salvar_modelo()