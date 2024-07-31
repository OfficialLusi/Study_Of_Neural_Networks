# train_model_1.py

# train a model with one input layer, 2 hidden layers and 1 output layer


import numpy as np
from keras import datasets  
from model_utils import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy, Optimizer_Adam, save_model

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalizza i dati
x_train = x_train / 255.0
x_test = x_test / 255.0

# Riformatta i dati: da (60000, 28, 28) a (60000, 784)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Definisci le etichette come array di interi
y_train = np.array(y_train)
y_test = np.array(y_test)

# Crea i livelli densi per il modello MNIST
dense1 = Layer_Dense(784, 256)  # Primo layer nascosto
activation1 = Activation_ReLU()
dense2 = Layer_Dense(256, 128)  # Secondo layer nascosto
activation2 = Activation_ReLU()
dense3 = Layer_Dense(128, 64)  # Layer di output
activation3 = Activation_ReLU()
dense4 = Layer_Dense(64, 10)  # Layer di output
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.001, decay=5e-7)

# Addestra la rete
epochs = 600 # Puoi aumentare il numero di epoche per un'accuratezza migliore
for epoch in range(epochs):
    # Forward pass
    dense1.forward(x_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    dense4.forward(activation3.output)
    loss = loss_activation.forward(dense4.output, y_train)

    # Predizioni e accuratezza
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_train.shape) == 2:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train
    accuracy = np.mean(predictions == y_train_labels)

    print(f"epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y_train)
    dense4.backward(loss_activation.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Aggiornamento dei parametri
    optimizer.pre_update_params()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.update_parameters(dense3)
    optimizer.update_parameters(dense4)
    optimizer.post_update_params()

path = "./"

# Salva i pesi del modello
save_model([dense1, dense2, dense3, dense4], 'model_2_10_epochs.json', path=path)
