import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoint/action/FT_ucf5_4/metric.csv')

def extract_tensor_value(tensor_str):
    return float(tensor_str.split('(')[1].split(',')[0])

df['train_top1'] = df['train_top1'].apply(extract_tensor_value)
df['test_top1'] = df['test_top1'].apply(extract_tensor_value)

best_epoch = df['test_top1'].idxmax()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(df['train_loss'], label='Train Loss', color='blue')
axs[0].plot(df['test_loss'], label='Test Loss', color='red')
axs[0].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch (Test Top-1: {df["test_top1"][best_epoch]:.2f}%)')
axs[0].set_title('Training and Testing Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(df['train_top1'], label='Train Top-1 Accuracy', color='b')
axs[1].plot(df['test_top1'], label='Test Top-1 Accuracy', color='r')
axs[1].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch (Test Top-1: {df["test_top1"][best_epoch]:.2f}%)')
axs[1].set_title('Training and Testing Top-1 Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Top-1 Accuracy (%)')
axs[1].legend()

plt.tight_layout()

plt.savefig('checkpoint/action/FT_ucf5_4/metric_graph.png')

# 그래프 화면에 출력 (옵션)
# plt.show()
