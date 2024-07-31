import os
import numpy as np
from keras import preprocessing, datasets
from model_utils import Layer_Dense, Activation_ReLU, Activation_Softmax, load_model
import matplotlib.pyplot as plt

models_path = [
    '2_models/model_1_500_epochs.json',
    '2_models/model_1_600_epochs.json', 
    '2_models/model_1_750_epochs.json',
    '2_models/model_1_1000_epochs.json',
]

images_path = [
    '1_images/0.png',
    '1_images/1.png',
    '1_images/2.png',
    '1_images/3.png',
    '1_images/4.png',
    '1_images/5.png',
    '1_images/6.png',
    '1_images/7.png',
    '1_images/8.png',
    '1_images/9.png'
]

def load_and_preprocess_image(image_path):
    # Carica l'immagine in scala di grigi e ridimensiona a 28x28 pixel
    img = preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    # Converti l'immagine in un array
    img_array = preprocessing.image.img_to_array(img)
    # Inverti i colori dell'immagine (MNIST ha lo sfondo nero e il testo bianco)
    img_array = 255 - img_array
    # Normalizza i valori dei pixel a [0, 1]
    img_array = img_array / 255.0
    # Riformatta l'array in un vettore di dimensione (784,)
    img_array = img_array.reshape(1, -1)

    return img_array



# Carica il dataset MNIST per avere i dati di test originali
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# Normalizza i dati di test originali
x_test = x_test / 255.0

# Riformatta i dati di test originali: da (10000, 28, 28) a (10000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)

# Definisci le etichette come array di interi
y_test = np.array(y_test)

# Crea i livelli densi per il modello MNIST
dense1 = Layer_Dense(784, 128)  # Primo layer nascosto
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)  # Secondo layer nascosto
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 10)  # Layer di output
activation_softmax = Activation_Softmax()

# Carica i pesi nei nuovi layer
load_model([dense1, dense2, dense3], models_path[2])

# Valutazione sul set di test con il modello caricato
dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation_softmax.forward(dense3.output)

predictions = np.argmax(activation_softmax.output, axis=1)
if len(y_test.shape) == 2:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test
accuracy = np.mean(predictions == y_test_labels)

print(f"Test accuracy (caricato): {accuracy:.3f}")

# Elenca tutte le immagini nella directory
# custom_images = [os.path.join(custom_images_path, img) for img in os.listdir(custom_images_path) if img.endswith('.png')]

images_reshaped = []
predictions = []

# Preprocessa e mostra le immagini personalizzate
for image_path in images_path:
    img_array = load_and_preprocess_image(os.path.join(image_path))

    # Valutazione sul set di test con il modello caricato
    dense1.forward(img_array)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation_softmax.forward(dense3.output)
    prediction = np.argmax(activation_softmax.output, axis=1)
    images_reshaped.append(img_array)
    predictions.append(prediction[0])
    

# Visualizza alcune predizioni del modello caricato
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images_reshaped[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.subplot(2, 5, i+1)
