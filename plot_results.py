import pandas as pd
import matplotlib.pyplot as plt

class TrainingResultsPlotter:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['epoch'], self.df['train/box_loss'], label='Train Box Loss')
        plt.plot(self.df['epoch'], self.df['val/box_loss'], label='Val Box Loss')
        plt.plot(self.df['epoch'], self.df['metrics/mAP_50-95'], label='mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Eğitim ve Doğrulama Kayıpları ile mAP')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    plotter = TrainingResultsPlotter('runs/detect/train6/results.csv')
    plotter.plot()

df = pd.read_csv('runs/detect/train6/results.csv')
print(df.columns)