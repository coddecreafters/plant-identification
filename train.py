import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import os
import gc

def train_model():
    # Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Define data directories
    train_dir = 'dataset/train'
    validation_dir = 'dataset/validation'
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,  # Increased batch size
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=16,  # Increased batch size
        class_mode='categorical'
    )

    # Create and compile model
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=5,  # Increased epochs for better accuracy with smaller model
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Clear memory
    del train_generator
    del validation_generator
    gc.collect()

    # Save the model
    model.save('plant_model.h5', save_format='h5')
    print("Model saved as 'plant_model.h5'")

if __name__ == '__main__':
    train_model()
