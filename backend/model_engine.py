import torch
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from PIL import Image
import io
import numpy as np
import os

class SegmentationEngine:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"🧠 Engine optimizing for: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}")

        # --- بخش اصلاح شده برای PyTorch 2.6 ---
        try:
            print("🔄 Attempting to load model (Method 1: Full Object)...")
            # تغییر مهم: weights_only=False اضافه شد تا ارور امنیتی رفع شود
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            print("✅ Model loaded successfully as Full Object.")
        except Exception as e1:
            print(f"⚠️ Method 1 failed ({e1}). Trying Method 2 (State Dict)...")
            try:
                # ساخت معماری خام
                self.model = smp.Unet(
                    encoder_name='efficientnet-b1',
                    encoder_weights=None,
                    in_channels=3,
                    classes=3
                )
                # لود کردن وزن‌ها با مجوز امنیتی
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                print("✅ Model loaded successfully via State Dict.")
            except Exception as e2:
                print("❌ CRITICAL: Both loading methods failed.")
                raise RuntimeError(f"Could not load model! Final Error: {e2}")

        self.model.to(self.device)
        self.model.eval()

        # پیش‌پردازش استاندارد
        self.transforms = v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).cpu().numpy()
                
            return prediction[0]
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise e