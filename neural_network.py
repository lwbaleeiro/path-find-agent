
# # Classe para a rede neural
# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         # Inicializa os pesos
#         self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
#         self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

#         # Inicializa os bias
#         self.bias_hidden = np.zeros((1, self.hidden_size))
#         self.bias_output = np.zeros((1, self.output_size))

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def sigmoid_derivative(self, x):
#         return x * (1 - x)

#     def forward(self, inputs):
#         self.hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
#         self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
#         return self.output

#     def backward(self, inputs, target, learning_rate):
#         output_error = target - self.output
#         output_delta = output_error * self.sigmoid_derivative(self.output)

#         hidden_error = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

#         # Atualiza os pesos e bias
#         self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
#         self.weights_input_hidden += inputs.reshape(-1, 1).dot(hidden_delta) * learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
#         self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = 0.01

        self.gamma = 0.9 # Fator de desconto para recompensas futuras
        self.epsilon = 1.0  # Taxa de exploração inicial
        self.epsilon_decay = 0.995  # Taxa de decaimento da taxa de exploração
        self.min_epsilon = 0.01  # Taxa de exploração mínima

        # Inicializa os pesos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Inicializa os bias
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, inputs, reward):

        expected_output = reward + self.gamma * np.max(self.output_size)

        output_error = expected_output - self.output_size
        output_delta = output_error * self.sigmoid_derivative(self.output_size)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_size)

        # Atualiza os pesos e bias
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

    def get_epsilon(self):
        return max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # Função para salvar e carregar a rede neural
    def save_model(self):
        np.savez("np_weights",
                weights_input_hidden = self.weights_input_hidden,
                weights_hidden_output = self.weights_hidden_output,
                bias_hidden = self.bias_hidden,
                bias_output = self.bias_output)

    def load_model(self):
        npzfile = np.load("np_weights")
        self.weights_input_hidden = npzfile['weights_input_hidden']
        self.weights_hidden_output = npzfile['weights_hidden_output']
        self.bias_hidden = npzfile['bias_hidden']
        self.bias_output = npzfile['bias_output']

    # Função para salvar os pesos da rede neural em um arquivo
    def save_weights(self):
        with open("weights.pkl", 'wb') as f:
            pickle.dump(self.weights_input_hidden, f)
            pickle.dump(self.weights_hidden_output, f)

    # Função para carregar os pesos da rede neural de um arquivo
    def load_weights(self):
        with open("weights.pkl", 'rb') as f:
            self.weights_input_hidden = pickle.load(f)
            self.weights_hidden_output = pickle.load(f)

# Exemplo de uso:
if __name__ == "__main__":
    nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=2)
    inputs = np.array([[0, 1, 0, 1]])
    expected_output = np.array([[1, 0]])

    output = nn.forward(inputs)
    print("Output antes do treinamento:", output)
    
    for _ in range(10000):  # Treina a rede
        nn.backward(inputs, expected_output, learning_rate=0.01)
        nn.forward(inputs)
    
    output = nn.forward(inputs)
    print("Output após o treinamento:", output)
