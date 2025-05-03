import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns

# Define path to the dataset
data_path = 'Animal_Sound_Dataset'

# Define fixed length for padding/trimming
max_length = 40  # or any other length that you want to set

# Initialize lists to store data and labels
data = []
labels = []

# Define path to save spectrogram images
spectrogram_path = 'Spectrograms'

# Create directory if it doesn't exist
if not os.path.exists(spectrogram_path):
    os.makedirs(spectrogram_path)

# Iterate through each folder (animal class)
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)

    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue

    spectrogram_folder_path = os.path.join(spectrogram_path, folder)

    # Create directory for spectrogram images
    if not os.path.exists(spectrogram_folder_path):
        os.makedirs(spectrogram_folder_path)

    for filename in os.listdir(folder_path):
        if not filename.endswith('.wav'):
            continue

        file_path = os.path.join(folder_path, filename)

        # Load audio file
        y, sr = librosa.load(file_path, duration=2.97)  # Adjust duration as needed

        # Generate and save spectrogram
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        image_path = os.path.join(spectrogram_folder_path, filename.replace('.wav', '.png').lower())
        plt.imsave(image_path, spectrogram, format='png')

        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Padding or trimming to fixed length
        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]

        # Append data and label
        data.append(mfccs)
        labels.append(folder)

# Convert lists to arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Expand dimensions to fit the input shape for CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert labels to categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('./model_checkpoint.keras', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train_categorical, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test_categorical),
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate training accuracy
train_loss, train_accuracy = model.evaluate(X_train, y_train_categorical)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Generate classification report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)

# Print classification report
print("Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Define the path to save the model
model_path = 'model_checkpoint.keras'

# Save the model
model.save(model_path)

print(f"Model saved at {model_path}")

import pickle

# Define the path to save the label encoder
encoder_path = 'label_encoder.pkl'

# Save the label encoder
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Label encoder saved at {encoder_path}")
