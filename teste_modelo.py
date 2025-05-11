import joblib

# Carregar o modelo salvo
modelo = joblib.load('outputs/modelo_final.joblib')

# Fazer uma previsão para uma temperatura de 30°C
temperatura = [[30]]  # Temperatura em graus Celsius
previsao = modelo.predict(temperatura)

# Exibir o resultado
print(f"Para uma temperatura de {temperatura[0][0]}°C, a previsão de vendas é de {int(previsao[0])} sorvetes.")