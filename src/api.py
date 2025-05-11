import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Criar app
app = FastAPI(title="API de Previsão de Vendas de Sorvete")

# Carregar modelo
modelo = joblib.load('outputs/modelo_final.joblib')

# Classe para dados de entrada
class TemperaturaInput(BaseModel):
    temperatura: float

# Endpoint para previsão
@app.post("/prever/")
def prever_vendas(dados: TemperaturaInput):
    # Formatar entrada para o modelo
    temperatura = [[dados.temperatura]]
    
    # Fazer previsão
    previsao = int(modelo.predict(temperatura)[0])
    
    # Retornar resultado
    return {
        "temperatura": dados.temperatura, 
        "previsao_vendas": previsao,
        "mensagem": f"Para uma temperatura de {dados.temperatura}°C, espera-se vender {previsao} sorvetes."
    }

# Endpoint de status
@app.get("/")
def status():
    return {"status": "online", "modelo": "vendas_sorvete_v1"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)