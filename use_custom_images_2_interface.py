import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize
import cv2
import numpy as np
from keras import preprocessing, datasets
from model_utils import Layer_Dense, Activation_ReLU, Activation_Softmax, load_model

models_path = ['2_models/model_2_600_epochs.json']

# images_path = [
#     '1_images/0.png',
#     '1_images/1.png',
#     '1_images/2.png',
#     '1_images/3.png',
#     '1_images/4.png',
#     '1_images/5.png',
#     '1_images/6.png',
#     '1_images/7.png',
#     '1_images/8.png',
#     '1_images/9.png',
# ]

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Carica ed Elabora Immagine")
        self.setGeometry(100, 100, 800, 600)

        # Crea i livelli densi per il modello MNIST
        self.dense1 = Layer_Dense(784, 256)  # Primo layer nascosto
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(256, 128)  # Secondo layer nascosto
        self.activation2 = Activation_ReLU()
        self.dense3 = Layer_Dense(128, 64)  # Layer di output
        self.activation3 = Activation_ReLU()
        self.dense4 = Layer_Dense(64, 10)  # Layer di output
        self.activation_softmax = Activation_Softmax()

        # Carica i pesi nei nuovi layer
        load_model([self.dense1, self.dense2, self.dense3, self.dense4], models_path[0])

        # Variabile per memorizzare il percorso dell'immagine caricata
        self.file_path = None

        # Layout principale
        grid_layout = QGridLayout()

        # Layout verticale per i pulsanti
        button_layout = QVBoxLayout()

        # Pulsante per caricare l'immagine
        self.load_button = QPushButton("Carica Immagine")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        # Pulsante per elaborare l'immagine
        self.edit_button = QPushButton("Elabora Immagine")
        self.edit_button.clicked.connect(self.process_image)
        button_layout.addWidget(self.edit_button)

        # Pulsante per eliminare l'immagine
        self.clear_button = QPushButton("Elimina Immagine")
        self.clear_button.clicked.connect(self.clear_image)
        button_layout.addWidget(self.clear_button)

        # Pulsante per fare la previsione
        self.predict_button = QPushButton("Prevedi Immagine")
        self.predict_button.clicked.connect(self.predict_image)
        button_layout.addWidget(self.predict_button)

        # Widget contenitore per il layout verticale
        button_container = QWidget()
        button_container.setLayout(button_layout)

        # Aggiungi il layout verticale alla griglia con allineamento in alto a sinistra
        grid_layout.addWidget(button_container, 0, 0, 1, 2, alignment=Qt.AlignTop | Qt.AlignCenter)

        # Etichetta per visualizzare l'immagine elaborata
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(QSize(400, 300))  # Dimensione fissa per un quarto della finestra (800x600)
        grid_layout.addWidget(self.image_label, 1, 0, 1, 2, alignment=Qt.AlignCenter)  # Span 2 columns

        # Etichetta per visualizzare il risultato della previsione
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.result_label, 2, 0, 1, 2, alignment=Qt.AlignCenter)  # Span 2 columns

        container = QWidget()
        container.setLayout(grid_layout)
        self.setCentralWidget(container)

    def load_image(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona un'immagine", "", "Image Files (*.png *.jpg *.bmp)")
        if self.file_path:
            img = cv2.imread(self.file_path)
            # Ridimensiona l'immagine a 28x28 pixel
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            # Converti l'immagine processata in un formato visualizzabile da QLabel
            height, width, channels = img.shape
            bytes_per_line = channels * width
            qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_image(self):
        if self.file_path:
            # Carica l'immagine usando OpenCV
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)

            # Ridimensiona l'immagine a 28x28 pixel
            resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

            # Applica una soglia per rendere l'immagine più nitida
            _, threshold_image = cv2.threshold(resized_image, 130, 255, cv2.THRESH_BINARY)

            # Converti l'immagine processata in un formato visualizzabile da QLabel
            height, width = threshold_image.shape
            qimage = QImage(threshold_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def predict_image(self):
        if self.file_path:
            img_array = load_and_preprocess_image(self.file_path)

            # Valutazione sul set di test con il modello caricato
            self.dense1.forward(img_array)
            self.activation1.forward(self.dense1.output)
            self.dense2.forward(self.activation1.output)
            self.activation2.forward(self.dense2.output)
            self.dense3.forward(self.activation2.output)
            self.activation3.forward(self.dense3.output)
            self.dense4.forward(self.activation3.output)
            self.activation_softmax.forward(self.dense4.output)
            prediction = np.argmax(self.activation_softmax.output, axis=1)
            
            predicted_digit = prediction[0]

            # Mostra il risultato della previsione
            self.result_label.setText(f"Predicted: {predicted_digit}")

    def clear_image(self):
        self.image_label.clear()  # Rimuove il contenuto dell'etichetta
        self.result_label.clear()  # Rimuove il contenuto dell'etichetta di risultato
        self.file_path = None  # Resetta il percorso del file caricato

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
