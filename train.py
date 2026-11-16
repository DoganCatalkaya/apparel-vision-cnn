import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Cihaz Seçimi (Ekran kartı varsa GPU, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# 2. Veri Hazırlığı ve Dönüşümler
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Eğitim verisini %80 Eğitim (48.000), %20 Doğrulama (12.000) olarak ayırma
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Basit CNN Model Mimarisi
class SimpleFashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleFashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Eğitim Döngüsü
EPOCHS = 10
print("\n--- Eğitim Başlıyor ---")
start_train_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Doğrulama (Validation)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Accuracy: %{val_acc:.2f}")

total_train_time = time.time() - start_train_time
print(f"\nToplam Eğitim Süresi: {total_train_time:.2f} saniye")

# 5. Test Seti Üzerinde Performans ve Tahmin Süresi Ölçümü
model.eval()
all_preds = []
all_labels = []

start_infer_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
total_infer_time = time.time() - start_infer_time
avg_infer_ms = (total_infer_time / len(test_dataset)) * 1000

acc = accuracy_score(all_labels, all_preds) * 100
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

print("\n--- TEST SONUÇLARI (Rapor Tablosuna Yazılacak Değerler) ---")
print(f"Test Accuracy : %{acc:.2f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"Tek Görüntü Tahmin Süresi (Inference Time): {avg_infer_ms:.4f} ms")

# 6. Modeli Kaydet
torch.save(model.state_dict(), "fashion_cnn.pth")
print("\nModel 'fashion_cnn.pth' olarak başarıyla kaydedildi!")