from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('classificador.pkl', 'rb') as file_classificador:
    classificador = pickle.load(file_classificador)

with open('vetorizador.pkl', 'rb') as file_vetorizador:
    vetorizador = pickle.load(file_vetorizador)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/classificador', methods=['POST'])
def classifier():
    frase = request.form['frase']

    # transformar a frase num array de número (vetorizador) - Bag of Words
    vetor = vetorizador.transform([frase])
    
    # realizar a predição no meu modelo classificador (Regressão Logística):
    predicao = classificador.predict(vetor)

    resultado = 'Frase negativa' if predicao[0] == 0 else 'Frase positiva'
    tipo = 'danger' if predicao[0] == 0 else 'success'

    return render_template('resultado.html', tipo=tipo, frase=frase, resultado=resultado)

# @app.route('/classifier/<frase>', methods=['GET'])
# def classifier(frase):
#     # transformar a frase num array de número (vetorizador) - Bag of Words
#     vetor = vetorizador.transform([frase])
    
#     # realizar a predição no meu modelo classificador (Regressão Logística):
#     predicao = classificador.predict(vetor)
    
#     if predicao == [0]:
#         return f'A frase "{frase}" é negativa.'
#     else:
#         return f'A frase "{frase}" é positiva.'

if __name__ == '__main__':
    app.run(debug=True)
