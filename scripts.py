from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados
(x_treino, y_treino), (x_teste, y_teste) = keras.datasets.cifar10.load_data()

# Normalização
x_treino = x_treino.astype("float32") / 255.0
x_teste = x_teste.astype("float32") / 255.0

# Flatten para MLP
x_treino = x_treino.reshape(-1, 32*32*3)
x_teste = x_teste.reshape(-1, 32*32*3)

# One-hot encoding
y_treino_categ = keras.utils.to_categorical(y_treino, 10)
y_teste_categ = keras.utils.to_categorical(y_teste, 10)

# Função para criar e treinar os modelos
def criar_e_treinar(arquitetura, ativacoes, nome):
    modelo = keras.Sequential()
    modelo.add(keras.layers.Dense(arquitetura[0], activation=ativacoes[0], input_shape=(32*32*3,)))
    
    for unidades, ativacao in zip(arquitetura[1:], ativacoes[1:]):
        modelo.add(keras.layers.Dense(unidades, activation=ativacao))
    
    modelo.add(keras.layers.Dense(10, activation='softmax'))

    modelo.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    print(f"Treinando modelo: {nome}")
    historico = modelo.fit(x_treino, y_treino_categ,
                           epochs=10, batch_size=128,
                           validation_split=0.1, verbose=1)
    
    y_predito = np.argmax(modelo.predict(x_teste), axis=1)
    y_verdadeiro = y_teste.flatten()
    
    matriz_confusao = confusion_matrix(y_verdadeiro, y_predito)
    relatorio = classification_report(y_verdadeiro, y_predito, digits=4)
    
    print(f"\n=== {nome} ===")
    print(relatorio)
    
    # Exibir matriz de confusão
    plt.figure(figsize=(8,6))
    plt.imshow(matriz_confusao, cmap='Blues')
    plt.title(f"Matriz de Confusão: {nome}")
    plt.colorbar()
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Verdadeira")
    plt.show()
    
    return matriz_confusao, relatorio

# Lista de classificadores
classificadores = [
    {"arquitetura": [512], "ativacoes": ["relu"], "nome": "MLP_1camada_relu"},
    {"arquitetura": [256, 128], "ativacoes": ["relu", "relu"], "nome": "MLP_2camadas_relu"},
    {"arquitetura": [1024, 512, 256], "ativacoes": ["relu", "relu", "relu"], "nome": "MLP_3camadas_relu"},
    {"arquitetura": [512, 256], "ativacoes": ["tanh", "tanh"], "nome": "MLP_2camadas_tanh"},
    {"arquitetura": [512, 256], "ativacoes": ["sigmoid", "sigmoid"], "nome": "MLP_2camadas_sigmoid"},
]

# Treinar e salvar resultados
resultados = []
for clf in classificadores:
    matriz, relatorio = criar_e_treinar(clf["arquitetura"], clf["ativacoes"], clf["nome"])
    resultados.append({"nome": clf["nome"], "matriz_confusao": matriz, "relatorio": relatorio})
