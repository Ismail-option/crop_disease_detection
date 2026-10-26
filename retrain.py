import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Settings
DATASET_PATH = r"C:\Users\Admin\crop_disease_detection\dataset\plantvillage_dataset\color"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # fewer epochs since model already knows a lot

# Load already trained model
print("Loading existing model...")
model = load_model('crop_disease_model.h5')
print("✅ Model loaded successfully!")

# Prepare data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Continue training
print("Continuing training with new data...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save updated model
model.save('crop_disease_model.h5')
print("✅ Updated model saved!")

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy After Retraining')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss After Retraining')
plt.legend()

plt.savefig('retraining_results.png')
plt.show()
print("✅ Retraining complete!")