import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# Sınıf İsimleri
CLASSES = [
    "T-shirt / Üst (T-shirt/top)",
    "Pantolon (Trouser)",
    "Kazak (Pullover)",
    "Elbise (Dress)",
    "Kaban / Ceket (Coat)",
    "Sandalet (Sandal)",
    "Gömlek (Shirt)",
    "Spor Ayakkabı (Sneaker)",
    "Çanta (Bag)",
    "Bot (Ankle boot)"
]

# Model Mimarisi (train.py ile birebir aynı)
class SimpleFashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_clothing(image, invert_colors):
    if image is None:
        return None, None
    
    # 1. Gri tonlamaya çevir
    gray_img = image.convert("L")
    
    # 2. Arka plan açıksa (beyaz duvar/çarşaf vb.) renkleri ters çevir (Fashion-MNIST siyah arka planlıdır)
    if invert_colors:
        gray_img = ImageOps.invert(gray_img)
        
    # 3. Kontrastı artır ki kıyafet netleşsin
    gray_img = ImageOps.autocontrast(gray_img)
    
    # Modelin gördüğü 28x28 hali (rapora koymak için görselleştiriyoruz)
    processed_preview = gray_img.resize((140, 140), Image.NEAREST)
    
    # Tensöre çevir ve tahmin yap
    tensor_img = transform(gray_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        
    # En yüksek 3 tahmini sözlük olarak döndür
    confidences = {CLASSES[i]: float(probs[i]) for i in range(10)}
    return confidences, processed_preview

# Gradio Arayüzü
interface = gr.Interface(
    fn=predict_clothing,
    inputs=[
        gr.Image(type="pil", label="Kıyafet Fotoğrafı Yükle"),
        gr.Checkbox(value=True, label="Açık Arka Plan (Fotoğrafı beyaz/açık zeminde çektiyseniz işaretli kalsın)")
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Model Tahminleri"),
        gr.Image(label="Modelin Gördüğü İşlenmiş Görüntü (28x28)")
    ],
    title="Kıyafet Sınıflandırma Sistemi (Fashion-MNIST CNN)",
    description="Telefonla çektiğiniz kıyafet fotoğrafını yükleyin. Model kıyafetin türünü tahmin edecektir."
)

if __name__ == "__main__":
    interface.launch()