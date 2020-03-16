##########################################################################################
# Importação das bibliotecas
##########################################################################################

import numpy as np

##########################################################################################
# Criando algumas funções que serão usadas no programa
##########################################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid):
    return sigmoid * (1 - sigmoid)

def error(x, y, method):
    if method == 'SIMPLES':
        return x - y
    if method == 'SE':
        return np.power(x - y, 2)
    
##########################################################################################
# Definição dos parâmetros globais
##########################################################################################

lr = 0.3
momentum = 1.0
num_epochs = 10000

# Dados de entrada
input_data = np.array([[0,0], 
                       [0,1], 
                       [1,0], 
                       [1,1]])

# Dados de saída (variável resposta)
output_data = np.array([[0], 
                        [1], 
                        [1], 
                        [0]])

# Pesos da camada de entrada para a camada oculta
# A quantidade de pesos que devemos criar será dada pela:
# "quantidade de neuronios de entrada" x "quantidade de neuronios da próxima camada"
# Ou seja, se temos 2 neurônios na camada de entrada e 3 neurônios na camada de saída, 
# teremos que criar 6 valores de pesos. Esses pesos serão armazenados em uma matriz de
# "quantidade de neurônios de entrada" x "quantidade de neurônios da próxima camada"
# Neste mesmo exemplo, seria uma matriz de 2 x 3 (2 linhas e 3 colunas)

w0 = np.array([[-0.424, -0.740, -0.961], 
               [0.358, -0.577, -0.469]])

# Também podemos fazer esta inicialização de forma aleatória, o que é o mais comum
w0 = 2 * np.random.random((2,3)) - 1

# Pesos da camada oculta para a camada de saída
# Segue-se a mesmo lógica. Se temos 3 neurônios na camada oculta e 1 neurônio na
# camada de saída, temos criar uma matriz de 3 x 1

w1 = np.array([[-0.017], 
               [-0.893], 
               [0.148]])

# Também podemos fazer esta inicialização de forma aleatória, o que é o mais comum
w1 = 2 * np.random.random((3,1)) - 1

##########################################################################################
# A partir deste momento as operações são feitas para cada epoch
# Uma epoch representa uma passagem completa por todos os registros do conjunto de 
# dados de treino
##########################################################################################

for epoch in range(num_epochs):
    
    ##########################################################################################
    # Fazendo os cálculos da camada de entrada
    ##########################################################################################
    
    # Calculando SOMA(Xi * wi)
    
    # Neste primeiro momento, precisamos multiplicar cada valor de entrada com cada valor de peso 
    # (em seus respectivos neurônios) e no fim somar estes valores. Esta etapa é dada pela fórmula:
    # SOMA(Xi * wi). O numpy possui um método .dot para realizar esta multiplicação e soma
    
    # Algumas regras sobre operação entre matrizes que podem lhe ajudar: 
    # Levando em consideração este novo caso, no qual temos a multiplicação de matrizes 4x2 e 2x3, considere que:
    #    - Aqueles números INTERNOS que representam a quantidade de colunas na matriz 1 e a 
    #    quantidade de linhas da matriz 2 devem ser iguais, caso contrário não será possível fazer
    #    a operação (neste caso, 2 e 2)
    #    - O resultado da operação será uma matriz com os números de fora que você vê ali em cima (4 x 3)
    # Use estas regras para facilitar seu entendimento e validar suas operações
    
    L0_sum = input_data.dot(w0)
    
    # Note que L0_sum possui 4 linhas e 3 colunas. 4 linhas representando cada registro da base de entrada
    # e 3 colunas representando cada neurônio da camada oculta
    
    
    # Aplicando a função de ativação
    L0_activation = sigmoid(L0_sum)
    
    ##########################################################################################
    # Fazendo os cálculos da camada de saída e erro
    ##########################################################################################
    
    # Agora temos como entrada o matriz gerada em L0_activation. Faremos a multiplicação desta
    # matriz com os pesos da camada oculta
    L1_sum = L0_activation.dot(w1)
    
    # Em seguida, aplicamos a função de ativação
    L1_activation = sigmoid(L1_sum)
    
    # Como já estamos na última camada, podemos calcular o erro
    L1_error = error(x = output_data, y = L1_activation, method = 'SIMPLES')
    mean_error = np.mean(np.abs(L1_error))
    print('O erro médio é {}'.format(mean_error))
    
    # Existem outros formas dpara se calcular o erro, como MSE ou RMSE
    L1_error_squared = error(x = output_data, y = L1_activation, method = 'SE')
    MSE = np.mean(np.abs(L1_error_squared))
    RMSE =  np.sqrt(MSE)
    
    # Não vamos usar estes outros tipos de erro, estão aqui apenas para referência
    
    ##########################################################################################
    # Iniciando o processo de back propagation para atualização de pesos
    ##########################################################################################
    
    # Começaremos o processo de atualização dos pesos calculando a derivada da função de ativação da camada de saída
    # # Note que:
    #   - cada função de ativação terá sua própria equação da derivada parcial
    L1_derivative = sigmoid_derivative(L1_activation)
    
    # Em seguida calculamos o delta da camada de saída (erro * derivada da camada de saída)
    L1_delta = L1_error * L1_derivative
    
    # E também temos que calcular o delta da camada oculta (delta saída * peso * derivada da camada de entrada)
    L0_derivative = sigmoid_derivative(L0_activation)
    step1 = L1_delta.dot(w1.T)
    L0_delta = step1 * L0_derivative
    
    # Agora podemos atualizar os pesos da camada oculta para a camada de saída
    # O próximo peso (n+1) será dado pela seguinte fórmula
    # peso_n+1 = (peso_n * momentum) + (entrada * delta * lr)
    step1 = w1 * momentum
    step2 = L0_activation.T.dot(L1_delta) * lr
    w1 = step1 + step2
    
    # Por fim, faremos a atualização dos pesos da camada de entrada para a camada oculta
    step1 = w0 * momentum
    step2 = input_data.T.dot(L0_delta) * lr
    w0 = step1 + step2
    
    
    
    