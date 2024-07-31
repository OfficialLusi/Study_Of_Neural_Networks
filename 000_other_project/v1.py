import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QLabel, QTextEdit, QVBoxLayout
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize
import cv2
import numpy as np
import keras

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Carica ed Elabora Immagine")
        self.setGeometry(100, 100, 800, 600)

        # Carica il modello Keras
        self.model = keras.models.load_model('./000_other_project/mymodels/mymodel_cnn.keras')

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
            img  = keras.preprocessing.image.load_img(self.file_path, color_mode='grayscale', target_size=(28,28))
            img_arr = keras.preprocessing.image.img_to_array(img)
            img_arr = 255-img_arr
            img_arr = img_arr/255

            img_arr = np.expand_dims(img_arr, axis=0)
            prediction = self.model.predict(img_arr)
            predicted_digit = np.argmax(prediction)

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