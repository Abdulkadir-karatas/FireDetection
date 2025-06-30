import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

class FireDetectionTrainer:
    def __init__(self, data_yaml_path, model_size='n', epochs=100, batch_size=16):
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def initialize_model(self):
        self.model = YOLO(f'yolov8{self.model_size}.pt')

    def train(self):
        if self.model is None:
            self.initialize_model()
        results = self.model.train(
            data=self.data_yaml_path,
            epochs=50,
            batch=self.batch_size,
            imgsz=640,
            patience=15,
            weight_decay=0.001,
            save=True,
            device='0' if torch.cuda.is_available() else 'cpu',
            resume=True
        )
        return results

    def validate(self):
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
        results = self.model.val()
        return results

    def predict(self, image_path):
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
        results = self.model.predict(image_path)
        return results

if __name__ == "__main__":
    trainer = FireDetectionTrainer(
        data_yaml_path="data.yaml",
        model_size='n',  # 'n', 's', 'm', 'l', 'x' seçeneklerinden biri
        epochs=100,
        batch_size=16
    )
    trainer.train()
    trainer.validate()

# results.csv dosyasının yolunu gir
df = pd.read_csv('runs/detect/train6/results.csv')

plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Eğitim ve Doğrulama Kayıpları ile mAP')
plt.legend()
plt.grid()
plt.show()