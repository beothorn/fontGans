# Imports

import numpy as np 
from numpy import random
from matplotlib import pyplot as plt

# Função de desenho de resultado
def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10,10), nrows=m, ncols=n, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(1 - img.reshape((2,2)),cmap='Greys_r')
    return fig, axes

# Exemplos de faces
faces = [
        np.array([1,0,0,1]),
        np.array([0.9,0.1,0.2,0.8]),
        np.array([0.9,0.2,0.1,0.8]),
        np.array([0.8,0.1,0.2,0.9]),
        np.array([0.8,0.2,0.1,0.9]),
        ]

_ = view_samples(faces,1,4)
plt.show() # Exibe a matriz de faces exemplo em 1x4

# Exemplos de ruído
ruido = [random.randn(2,2) for i in range(20)]
print(ruido)

_ = view_samples(ruido, 4, 5)
plt.show() # Exibe a matriz de ruídos exemplo em 4x5

#
# Construido a rede neural
#

# A função de ativação sigmoid
def sigmoid(x):
    return np.exp(x)/(1.0+np.exp(x))

# Função reLu para avaliar
def relu(x):
    return (abs(x) + x) / 2

def leaky_relu(x):
    return np.where(x>0 , x, x * 0.01)

# Centralizar local da escolha de função de ativação
def activation(x):
    return sigmoid(x)

class Discriminator():
    def __init__(self):
        self.weights = np.array([random.normal() for i in range(4)])
        self.bias = random.normal()

    def forward(self, x):
        # Caminho para frente
        #Multiplica os valores da matriz com os pesos e soma o bias
        return activation(np.dot(x, self.weights)+self.bias)
        
    def error_from_image(self, image):
        prediction = self.forward(image)
        #Desejamos que a previsão seja 1, então o erro precisa ser -log(previsão)
        return -np.log(prediction)
        
    def derivatives_from_image(self, image):
        prediction = self.forward(image)
        derivatives_weights = -image * (1-prediction)
        derivative_bias = -(1-prediction)
        return derivatives_weights, derivative_bias

    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]
    
    def error_from_ruido(self, ruido):
        prediction = self.forward(ruido)
        # Desejamos que essa previsão seja 0, então o erro é -log(1-previsão)
        return -np.log(1-prediction)
    
    def derivatives_from_ruido(self, ruido):
        prediction = self.forward(ruido)
        derivatives_weights = ruido * prediction
        derivative_bias = prediction
        return derivatives_weights, derivative_bias

    def update_from_ruido(self, ruido):
        ders = self.derivatives_from_ruido(ruido)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]

class Generator():
    def __init__(self):
        self.weights = np.array([random.normal() for i in range(4)])
        self.biases = np.array([random.normal() for i in range(4)])
    
    def forward(self, z):
        #Repasse adiante
        return activation(z * self.weights + self.biases)

    def error(self, z, discriminator):
        x = self.forward(z)
        # Desejamos que a previsão seja 0, então o erro é -log(1-previsão)
        y = discriminator.forward(x)
        return -np.log(y)

    def derivatives(self, z , discriminator):
        discriminator_weights = discriminator.weights
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1-y) * discriminator_weights * x * (1-x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias
        
    def update(self, z, discriminator):
        error_before = self.error(z, discriminator) 
        ders = self.derivatives(z, discriminator)
        self.weights -= learning_rate * ders[0]
        self.biases -= learning_rate * ders[1]
        error_after = self.error(z, discriminator)

    
#
# Training
#

# Definir o seed fixo
#random.seed(420)

# Hiperparametros
learning_rate = 0.001
epochs = 26000

# A GAN
D = Discriminator()
G = Generator()

# Para plotar o erro
errors_discriminator = []
errors_generator = []

for epoch in range(epochs):
    for face in faces:
        # Atualizar os pesos do discriminador com uma face real
        D.update_from_image(face)
        # Pegar um número aleatório para gerar um rosto falso
        z = random.rand()
        # Calcular o erro do discriminador
        errors_discriminator.append(
            sum( D.error_from_image(face) + D.error_from_ruido(z) ) 
        )
        # Calcular o erro do gerador
        errors_generator.append(
            G.error(z,D)
        )
        # Construir um rosto falso
        fake_face = G.forward(z)
        # Atualizar os pesos do discriminador com o novo rosto falso
        D.update_from_ruido(fake_face)
        # Atualizar os pesos do gerador a partir do rosto falso
        G.update(z,D)
    
plt.plot(errors_generator)
plt.title("Generator error function")
plt.legend("gen")
plt.show()

plt.plot(errors_discriminator)
plt.title("Discriminator error function")
plt.legend("disc")
plt.show()

#
# Gerando imagens
#
generated_images = []
for i in range(4):
    z = random.random()
    generated_image = G.forward(z)
    generated_images.append(generated_image)
_ = view_samples(generated_images, 1, 4)
plt.show()

for i in generated_images:
    print(i)

#
# Estudo de bias
#
print("Generator weights", G.weights)
print("Generator biases", G.biases)

print("Discriminator weights", D.weights)
print("Discriminator bias", D.bias)