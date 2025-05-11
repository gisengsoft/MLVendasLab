import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def carregar_dados(caminho_arquivo):
    """Carrega os dados do arquivo CSV."""
    try:
        dados = pd.read_csv(caminho_arquivo)
        print(f"Dados carregados com sucesso: {dados.shape[0]} registros e {dados.shape[1]} colunas.")
        return dados
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def explorar_dados(dados):
    """Explora os dados e exibe informações básicas."""
    print("Primeiras 5 linhas:")
    print(dados.head())
    
    print("\nInformações dos dados:")
    print(dados.info())
    
    print("\nEstatísticas descritivas:")
    print(dados.describe())
    
    print("\nVerificando valores nulos:")
    print(dados.isnull().sum())
    
    print("\nCorrelação entre as variáveis:")
    print(dados[['Temperatura', 'Vendas']].corr())

def preparar_dados(dados, test_size=0.2, random_state=42):
    """Prepara os dados para treinamento e teste."""
    # Separar features e target
    X = dados[['Temperatura']].values
    y = dados['Vendas'].values
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Dados divididos: {X_train.shape[0]} amostras de treino e {X_test.shape[0]} amostras de teste.")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Testar as funções
    dados = carregar_dados("inputs/base_vendas_sorvete.csv")
    if dados is not None:
        explorar_dados(dados)
        X_train, X_test, y_train, y_test = preparar_dados(dados)
        print("Pré-processamento concluído com sucesso!")