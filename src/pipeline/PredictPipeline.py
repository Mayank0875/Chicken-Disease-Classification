import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.h5")
        self.img_size = (224, 224)

    def _load_image(self) -> np.ndarray:
        """Load and preprocess the image for prediction."""
        img = image.load_img(self.filename, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        return img_array

    def _load_model(self):
        """Load the trained Keras model from the predefined path."""
        return load_model(self.model_path)

    def predict(self) -> list[dict[str, str]]:
        """Run prediction and return the disease class."""
        model = self._load_model()
        img_array = self._load_image()
        prediction_idx = np.argmax(model.predict(img_array), axis=1)[0]


        label = "Healthy" if prediction_idx == 1 else "Coccidiosis"
        return [{"image": label}]
