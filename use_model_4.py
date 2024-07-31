# use_model.py
import numpy as np
from keras import datasets 
from model_utils import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy, load_model

models_path = [
    '2_models/model_4_400_epochs.json'
]
# Carica il dataset MNIST
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# Normalizza i dati
x_test = x_test / 255.0

# Riformatta i dati: da (10000, 28, 28) a (10000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)

# Definisci le etichette come array di interi
y_test = np.array(y_test)

# Crea i livelli densi per il modello MNIST
dense1 = Layer_Dense(784, 784)  # Primo layer nascosto
activation1 = Activation_ReLU()
dense2 = Layer_Dense(784, 512)  # Secondo layer nascosto
activation2 = Activation_ReLU()
dense3 = Layer_Dense(512, 256)  # Layer di output
activation3 = Activation_ReLU()
dense4 = Layer_Dense(256, 128)  # Layer di output
activation4 = Activation_ReLU()
dense5 = Layer_Dense(128, 64)  # Layer di output
activation5 = Activation_ReLU()
dense6 = Layer_Dense(64, 10)  # Layer di output
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# Carica i pesi nei nuovi layer
load_model([dense1, dense2, dense3, dense4, dense5, dense6], models_path[0])

# Valutazione sul set di test con il modello caricato
dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
dense4.forward(activation3.output)
activation4.forward(dense4.output)
dense5.forward(activation4.output)
activation5.forward(dense5.output)
dense6.forward(activation5.output)
loss = loss_activation.forward(dense6.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test
accuracy = np.mean(predictions == y_test_labels)

print(f"Test accuracy (caricato): {accuracy:.3f}, Test loss (caricato): {loss:.3f}")

# Visualizza alcune predizioni del modello caricato
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
