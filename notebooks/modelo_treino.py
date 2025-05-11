# %% [markdown]
# # Treinamento do Modelo de Previsão de Vendas de Sorvete
# 
# Este notebook demonstra o processo completo de análise de dados e treinamento do modelo para a sorveteria Gelato Mágico.

# %%
# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import mlflow

# Adicionar diretório src ao path para importar nossos módulos
sys.path.append('../')
from src.pre_processamento import carregar_dados, explorar_dados, preparar_dados
from src.modelo import ModeloVendasSorvete

# %% [markdown]
# ## 1. Carregamento e Exploração dos Dados

# %%
# Carregar os dados
caminho_dados = '../inputs/base_vendas_sorvete.csv'
dados = carregar_dados(caminho_dados)

# %%
# Exibir informações básicas dos dados
print("Primeiras 5 linhas:")
print(dados.head())

print("\nInformações dos dados:")
dados.info()

# %%
# Estatísticas descritivas
dados.describe()

# %% [markdown]
# ## 2. Análise Exploratória dos Dados

# %%
# Verificar a correlação entre temperatura e vendas
correlacao = dados[['Temperatura', 'Vendas']].corr()
print("Correlação entre Temperatura e Vendas:")
print(correlacao)

# Visualizar correlação
plt.figure(figsize=(10, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

# %%
# Gráfico de dispersão entre Temperatura e Vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperatura', y='Vendas', data=dados)
plt.title('Relação entre Temperatura e Vendas de Sorvete')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Distribuição da temperatura
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.histplot(dados['Temperatura'], kde=True)
plt.title('Distribuição da Temperatura')
plt.xlabel('Temperatura (°C)')

plt.subplot(1, 2, 2)
sns.histplot(dados['Vendas'], kde=True)
plt.title('Distribuição das Vendas')
plt.xlabel('Vendas de Sorvete')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Preparação dos Dados

# %%
# Preparar dados para treinamento e teste
X_train, X_test, y_train, y_test = preparar_dados(dados)

print(f"Formas dos conjuntos:\nX_train: {X_train.shape}\nX_test: {X_test.shape}\ny_train: {y_train.shape}\ny_test: {y_test.shape}")

# %% [markdown]
# ## 4. Treinamento do Modelo

# %%
# Criar e treinar o modelo
modelo = ModeloVendasSorvete()
modelo.treinar(X_train, y_train)

# %% [markdown]
# ## 5. Avaliação do Modelo

# %%
# Fazer previsões com o conjunto de teste
y_pred = modelo.prever(X_test)

# Avaliar o modelo
metricas = modelo.avaliar(X_test, y_test)

# Exibir métricas
for metrica, valor in metricas.items():
    print(f"{metrica}: {valor:.2f}")

# %%
# Visualizar os resultados
modelo.visualizar_resultados(
    np.vstack((X_train, X_test)), 
    np.concatenate((y_train, y_test)),
    X_test, y_test, y_pred
)

# %% [markdown]
# ## 6. Registro com MLflow

# %%
# Registrar modelo no MLflow
run_id = modelo.registrar_modelo_mlflow(X_train, y_train, X_test, y_test, run_name="Notebook_Run")
print(f"Modelo registrado no MLflow com run_id: {run_id}")

# %% [markdown]
# ## 7. Usando o Modelo para Previsões

# %%
# Fazer previsões para diferentes temperaturas
temperaturas_demo = np.arange(20, 36, 1).reshape(-1, 1)
vendas_previstas = modelo.prever(temperaturas_demo)

# Criar DataFrame com resultados
demo_df = pd.DataFrame({
    'Temperatura': temperaturas_demo.flatten(),
    'Vendas_Previstas': vendas_previstas.astype(int)
})

# Exibir resultados
print(demo_df)

# Plotar as previsões
plt.figure(figsize=(10, 6))
plt.plot(demo_df['Temperatura'], demo_df['Vendas_Previstas'], 'r-', linewidth=2)
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Dados de Teste')
plt.title('Previsão de Vendas para Diferentes Temperaturas')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %%
# Demonstração de uso prático: insira uma temperatura para prever vendas
def prever_vendas(temperatura):
    previsao = modelo.prever(temperatura)
    return int(previsao[0])

# Exemplo de uso
temp_exemplo = 30
vendas_previstas = prever_vendas(temp_exemplo)
print(f"Para uma temperatura de {temp_exemplo}°C, a previsão de vendas é de {vendas_previstas} sorvetes.")

# %% [markdown]
# ## 8. Salvando o Modelo para Uso Futuro

# %%
# Salvar o modelo treinado
modelo.salvar_modelo(caminho='../outputs/modelo_final.joblib')

# Demonstração: como carregar o modelo salvo
novo_modelo = ModeloVendasSorvete()
novo_modelo.carregar_modelo(caminho='../outputs/modelo_final.joblib')

# Testar o modelo carregado
teste_temp = 25
previsao = int(novo_modelo.prever(teste_temp)[0])
print(f"Teste com modelo carregado: Para {teste_temp}°C, previsão de {previsao} vendas.")