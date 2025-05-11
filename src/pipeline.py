import os
import sys
import logging
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pre_processamento import carregar_dados, explorar_dados, preparar_dados
from modelo import ModeloVendasSorvete

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def executar_pipeline(caminho_dados, test_size=0.2, random_state=42):
    """
    Executa o pipeline completo de treinamento, avaliação e registro do modelo.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV de dados
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória para reprodutibilidade
    """
    try:
        # Criar diretório de saída se não existir
        os.makedirs('outputs', exist_ok=True)
        
        # 1. Carregar dados
        logger.info("Carregando dados...")
        dados = carregar_dados(caminho_dados)
        if dados is None:
            logger.error("Falha ao carregar os dados. Pipeline encerrado.")
            return
        
        # 2. Explorar dados
        logger.info("Explorando dados...")
        explorar_dados(dados)
        
        # 2.1 Salvar visualizações dos dados
        logger.info("Criando visualizações dos dados...")
        plt.figure(figsize=(16, 10))
        
        # Histograma da temperatura
        plt.subplot(2, 2, 1)
        sns.histplot(dados['Temperatura'], kde=True)
        plt.title('Distribuição da Temperatura')
        plt.xlabel('Temperatura (°C)')
        
        # Histograma das vendas
        plt.subplot(2, 2, 2)
        sns.histplot(dados['Vendas'], kde=True)
        plt.title('Distribuição das Vendas')
        plt.xlabel('Vendas de Sorvete')
        
        # Boxplot da temperatura
        plt.subplot(2, 2, 3)
        sns.boxplot(x=dados['Temperatura'])
        plt.title('Boxplot da Temperatura')
        plt.xlabel('Temperatura (°C)')
        
        # Boxplot das vendas
        plt.subplot(2, 2, 4)
        sns.boxplot(x=dados['Vendas'])
        plt.title('Boxplot das Vendas')
        plt.xlabel('Vendas de Sorvete')
        
        plt.tight_layout()
        plt.savefig('outputs/distribuicao_dados.png')
        
        # Visualizar correlação
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Temperatura', y='Vendas', data=dados)
        plt.title('Correlação entre Temperatura e Vendas')
        plt.xlabel('Temperatura (°C)')
        plt.ylabel('Vendas de Sorvete')
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/correlacao.png')
        
        # 3. Preparar dados
        logger.info("Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = preparar_dados(
            dados, test_size=test_size, random_state=random_state
        )
        
        # 4. Criar e treinar modelo
        logger.info("Criando e treinando modelo...")
        modelo = ModeloVendasSorvete()
        
        # 5. Registrar modelo no MLflow
        logger.info("Registrando modelo no MLflow...")
        run_id = modelo.registrar_modelo_mlflow(X_train, y_train, X_test, y_test)
        
        # 6. Salvar modelo
        logger.info("Salvando modelo treinado...")
        modelo.salvar_modelo()
        
        # 7. Demonstração de uso do modelo
        logger.info("Demonstração de uso do modelo:")
        
        # Previsões para temperatura variando de 20 a 35 graus
        temperaturas_demo = np.arange(20, 36, 1).reshape(-1, 1)
        vendas_previstas = modelo.prever(temperaturas_demo)
        
        # Criar DataFrame com resultados
        demo_df = pd.DataFrame({
            'Temperatura': temperaturas_demo.flatten(),
            'Vendas_Previstas': vendas_previstas.astype(int)
        })
        
        logger.info("\nPrevisões de vendas para diferentes temperaturas:")
        logger.info(demo_df)
        
        # Salvar tabela de demonstração
        demo_df.to_csv('outputs/previsoes_demonstracao.csv', index=False)
        
        # 8. Concluir pipeline
        logger.info("Pipeline executado com sucesso!")
        logger.info(f"Verifique os resultados na pasta 'outputs' e no MLflow (run_id: {run_id})")
        
        return modelo, run_id
    
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        return None, None

if __name__ == "__main__":
    # Executar pipeline
    executar_pipeline('inputs/base_vendas_sorvete.csv')