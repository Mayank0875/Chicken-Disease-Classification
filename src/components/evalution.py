import tensorflow as tf
import os
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationConfig:
    model_path: str = "artifacts/training/model.h5"
    data_path: str = "artifacts/raw_data"
    scores_path: str = "artifacts/evaluation/scores.json"
    image_size: tuple = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.3


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


class ModelEvaluator:
    def __init__(self):
        self.config = EvaluationConfig()
        self.model = self._load_model()
        self.validation_generator = self._create_validation_generator()

    def _load_model(self):
        print("[INFO] Loading model...")
        return tf.keras.models.load_model(self.config.model_path)

    def _create_validation_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=self.config.validation_split
        )

        dataflow_kwargs = dict(
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        print("[INFO] Creating validation generator...")
        return valid_datagenerator.flow_from_directory(
            directory=self.config.data_path,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluate_and_save(self):
        print("[INFO] Evaluating model...")
        loss, acc = self.model.evaluate(self.validation_generator)
        print(f"[RESULT] Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        save_json(self.config.scores_path, {"loss": loss, "accuracy": acc})



if __name__ == "__main__":

    evaluator = ModelEvaluator()
    evaluator.evaluate_and_save()
