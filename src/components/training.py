import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    training_data_path: str = os.path.join("artifacts", "raw_data")  # Correct path to your training data
    validation_data_path: str = os.path.join("artifacts", "raw_data")  # Correct path to your validation data
    model_save_path: str = os.path.join("artifacts", "training", "model.h5")  # Path to save the trained model
    image_size: tuple = (224, 224, 3)
    batch_size: int = 16
    epochs: int = 5
    classes: int = 2  # Number of classes, update accordingly


class SimpleCNN:
    def __init__(self, ):
        self.config = TrainingConfig()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes for classification
    ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data_generators(self):
        # Image data generators for training and validation
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.config.training_data_path,
            target_size=self.config.image_size[:2],  # Exclude channels (3rd dimension)
            batch_size=self.config.batch_size,
            class_mode='categorical'
        )

        self.validation_generator = test_datagen.flow_from_directory(
            self.config.validation_data_path,
            target_size=self.config.image_size[:2],  # Exclude channels (3rd dimension)
            batch_size=self.config.batch_size,
            class_mode='categorical'
        )

    def train(self):
        self.prepare_data_generators()
        
        print(f"Training for {self.config.epochs} epochs...")
        self.model.fit(
            self.train_generator,
            epochs=self.config.epochs,
            validation_data=self.validation_generator
        )

        # Save the trained model
        self.model.save(self.config.model_save_path)
        print(f"Model saved to {self.config.model_save_path}")

if __name__ == "__main__":

    cnn = SimpleCNN()
    cnn.train()
