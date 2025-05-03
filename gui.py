import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from IPython.display import Audio, display
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import simpleaudio as sa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model and label encoder
model_path = 'model_checkpoint.keras'
label_encoder_path = 'label_encoder.pkl'

model = load_model(model_path)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Function to predict pet animal
def predict_pet_animal(audio_file_path):
    y, sr = librosa.load(audio_file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    max_length = 40
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    predictions = model.predict(np.array([mfccs]))
    predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]
    
    return predicted_label

# Create the main window
root = tk.Tk()
root.title("Pet Sound Recognition")

# Function to handle closing the window
def on_closing():
    root.destroy()  # Destroy the Tkinter window
    plt.close()     # Close the Matplotlib plot
    root.quit()     # Quit the main loop

# Bind the closing function to the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Add a label
label = tk.Label(root, text="Upload an audio file to predict the pet animal")
label.pack(pady=20)

# Create Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Function to display the spectrogram with higher quality
def display_spectrogram(audio_file_path):
    # Clear the previous spectrogram
    ax.clear()

    y, sr = librosa.load(audio_file_path, duration=2.97)
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512, window='hann')), ref=np.max)
    spec_img = ax.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, spectrogram.shape[1], 0, sr // 2])
    ax.set_title('Spectrogram')

    # Add or update the colorbar
    if not hasattr(display_spectrogram, 'colorbar'):
        display_spectrogram.colorbar = fig.colorbar(spec_img, ax=ax, format='%+2.0f dB')

    # Update the canvas
    canvas.draw()

# Function to handle file upload and prediction
def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("Audio files", "*.wav")])
    if file_path:
        display(Audio(file_path))
        display_spectrogram(file_path)  # Display the spectrogram

# Function to play the uploaded sound
def play_sound():
    if 'file_path' not in globals():
        messagebox.showwarning("Warning", "No audio file uploaded yet!")
        return
    
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Function to get prediction based on the uploaded sound
def get_prediction():
    if 'file_path' not in globals():
        messagebox.showwarning("Warning", "No audio file uploaded yet!")
        return
    
    predicted_animal = predict_pet_animal(file_path)
    messagebox.showinfo("Prediction", f"The predicted pet animal is: {predicted_animal}")

# Add a button to upload file
upload_button = tk.Button(root, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=10)

# Add a button to play the uploaded sound
play_button = tk.Button(root, text="Play Uploaded Sound", command=play_sound)
play_button.pack(pady=10)

# Add a button to get prediction
prediction_button = tk.Button(root, text="Get Prediction", command=get_prediction)
prediction_button.pack(pady=10)

# Run the GUI
root.mainloop()
