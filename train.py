import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import os
import gc

def train_model():
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
        batch_size=32,  # Increased batch size for faster training
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Create and compile model
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,  # More epochs to compensate for smaller model
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Clear memory
    del train_generator
    del validation_generator
    gc.collect()

    # Save the model with minimal size
    model.save('plant_model.h5', save_format='h5', include_optimizer=False)
    print("Model saved as 'plant_model.h5'")

if __name__ == '__main__':
    train_model()
