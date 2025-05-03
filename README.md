# Sound-Based-Pets-Animal-Recognition-System

This project uses a Convolutional Neural Network (CNN) to classify animal sounds, specifically targeting pet animals. It includes scripts for training the model on an audio dataset and a graphical user interface (GUI) for predicting the animal from a new audio file.

## Features

* **Audio Feature Extraction:** Extracts Mel-frequency cepstral coefficients (MFCCs) from `.wav` audio files.
* **Spectrogram Generation:** Creates and saves spectrogram images for visualization.
* **CNN Model Training:** Trains a CNN model using TensorFlow/Keras to classify animal sounds based on MFCC features.
* **Model Evaluation:** Evaluates the trained model using accuracy, confusion matrix, and classification report.
* **GUI Application:** Provides a simple Tkinter-based GUI to:
    * Upload `.wav` audio files.
    * Display the spectrogram of the uploaded audio.
    * Play the uploaded audio.
    * Predict the animal type using the trained model.

## Dataset

* The training script (`main.py`) expects a dataset located in a folder named `Animal_Sound_Dataset` in the same directory as the script.
* Inside `Animal_Sound_Dataset`, create subfolders for each animal class (e.g., `Dog`, `Cat`, `Bird`).
* Place the corresponding `.wav` audio files for each animal into its respective subfolder.

## Dependencies

You'll need the following Python libraries:

* numpy
* librosa
* matplotlib
* scikit-learn
* tensorflow
* seaborn
* simpleaudio
* tk (usually included with Python)

## Installation

1.  **Clone the repository or download the files.**
2.  **Create the dataset folder:** Make sure you have the `Animal_Sound_Dataset` folder structured as described above.
3.  **Install dependencies:**
    ```bash
    pip install numpy librosa matplotlib scikit-learn tensorflow seaborn simpleaudio
    ```
    *(Note: Tkinter is typically included with standard Python installations.)*

## Usage

### 1. Training the Model

* Run the `main.py` script:
    ```bash
    python main.py
    ```
* This script will:
    * Process the audio files in `Animal_Sound_Dataset`.
    * Generate and save spectrograms in the `Spectrograms/` folder.
    * Train the CNN model.
    * Print evaluation metrics (accuracy, classification report) and display plots (confusion matrix, accuracy history).
    * Save the trained model as `model_checkpoint.keras`.
    * Save the label encoder as `label_encoder.pkl`.

### 2. Running the GUI Application

* Ensure the `model_checkpoint.keras` and `label_encoder.pkl` files (generated during training) are in the same directory.
* Run the `gui.py` script:
    ```bash
    python gui.py
    ```
* The GUI window will appear:
    * Click **"Upload Audio File"** to select a `.wav` file. The spectrogram will be displayed.
    * Click **"Play Uploaded Sound"** to listen to the selected audio.
    * Click **"Get Prediction"** to classify the animal sound using the trained model. The prediction will be shown in a message box.

## File Structure
```
Sound-Based-Pets-Animal-Recognition-System
├── Animal_Sound_Dataset/ # Input dataset (User needs to create this)
│   ├── Dog/
│   │   ├── dog_bark1.wav
│   │   └── ...
│   ├── Cat/
│   │   ├── cat_meow1.wav
│   │   └── ...
│   └── ...
├── Spectrograms/         # Generated spectrogram images (Output of main.py)
│   ├── Dog/
│   │   ├── dog_bark1.png
│   │   └── ...
│   ├── Cat/
│   │   ├── cat_meow1.png
│   │   └── ...
│   └── ...
├── main.py               # Script for data processing, training, and evaluation
├── gui.py                # Script for the GUI application
├── model_checkpoint.keras # Saved trained model (Output of main.py)
└── label_encoder.pkl     # Saved label encoder (Output of main.py)
```
## Model Details

* **Features:** 40 Mel-frequency cepstral coefficients (MFCCs) extracted from audio segments.
* **Architecture:** Convolutional Neural Network (CNN) implemented using TensorFlow/Keras. The specific layers include Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
