import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(num_classes):
    model = models.Sequential([
        # Single Convolutional Block with minimal filters
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((4, 4)),  # Aggressive pooling to reduce size
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
