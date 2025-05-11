import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Função para gerar dados sintéticos
def gerar_dados_sinteticos(n_samples=100):
    # Dados iniciais da imagem
    dados_iniciais = {
        'Data': ['01/01/2025', '02/01/2025', '03/01/2025', '04/01/2025', '05/01/2025', 
                '06/01/2025', '07/01/2025', '08/01/2025', '09/01/2025'],
        'Vendas': [120, 150, 100, 180, 130, 170, 140, 160, 110],
        'Temperatura': [30, 32, 28, 33, 29, 31, 30, 32, 27]
    }
    
    df_inicial = pd.DataFrame(dados_iniciais)
    
    # Se já temos 100 amostras, retornamos
    if len(df_inicial) >= n_samples:
        return df_inicial.iloc[:n_samples]
    
    # Caso contrário, geramos mais dados
    # Primeiro, vamos extrair a relação entre temperatura e vendas
    # para manter a correlação observada nos dados originais
    temperaturas = np.arange(20, 38)  # Variação de temperatura de 20 a 37 graus
    
    # Adicionar uma variação aleatória baseada na relação observada:
    # Da imagem observamos aproximadamente: Vendas ≈ 10*Temperatura - 180 + ruído
    vendas = [max(10 * temp - 180 + np.random.normal(0, 15), 50) for temp in temperaturas]
    vendas = [int(round(v)) for v in vendas]  # Arredondamos para números inteiros
    
    # Gerar datas sequenciais a partir da última data
    ultima_data = datetime.strptime(df_inicial['Data'].iloc[-1], '%d/%m/%Y')
    datas = [(ultima_data + timedelta(days=i+1)).strftime('%d/%m/%Y') 
             for i in range(n_samples - len(df_inicial))]
    
    # Gerar temperaturas e vendas correspondentes
    np.random.seed(42)  # Para reprodutibilidade
    temps_adicionais = np.random.choice(temperaturas, size=n_samples - len(df_inicial))
    vendas_adicionais = [vendas[np.where(temperaturas == t)[0][0]] + np.random.randint(-10, 11) 
                         for t in temps_adicionais]
    
    # Criar DataFrame adicional
    df_adicional = pd.DataFrame({
        'Data': datas,
        'Temperatura': temps_adicionais,
        'Vendas': vendas_adicionais
    })
    
    # Concatenar com os dados originais
    df_completo = pd.concat([df_inicial, df_adicional], ignore_index=True)
    
    # Garantir que temos exatamente n_samples
    return df_completo.iloc[:n_samples]

if __name__ == "__main__":
    # Gerar 100 amostras
    df_completo = gerar_dados_sinteticos(100)
    
    # Salvar no arquivo CSV
    df_completo.to_csv('inputs/base_vendas_sorvete.csv', index=False)
    print(f"Dataset gerado com {len(df_completo)} amostras e salvo em 'inputs/base_vendas_sorvete.csv'")