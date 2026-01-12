from fastapi import FastAPI, UploadFile, File
from model_engine import SegmentationEngine
import uvicorn
import os
import sys

app = FastAPI()

# --- بخش اصلاح شده برای پیدا کردن مسیر دقیق مدل ---
# 1. پیدا کردن مسیری که همین فایل main.py توش هست
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. رفتن یک مرحله عقب‌تر و ورود به پوشه checkpoints
model_path_abs = os.path.join(current_dir, "..", "checkpoints", "model.pt")
# 3. نرمال کردن مسیر (حذف .. و مرتب کردن اسلش‌ها برای ویندوز)
MODEL_PATH = os.path.normpath(model_path_abs)

print(f"📍 Looking for model at: {MODEL_PATH}")

engine = None

if os.path.exists(MODEL_PATH):
    try:
        print("⏳ Loading model... This might take a moment.")
        engine = SegmentationEngine(model_path=MODEL_PATH)
        print("✅ AI Engine Loaded Successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Model found but failed to load.")
        print(f"Error details: {e}")
        # نکته: اگر اینجا ارور گرفتی، یعنی فایل model.pt ناقص دانلود شده یا خرابه
else:
    print(f"❌ ERROR: File not found!")
    print(f"System expected file at: {MODEL_PATH}")

@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": engine is not None}

@app.post("/segment")
async def process_image(file: UploadFile = File(...)):
    if engine is None:
        return {
            "status": "error", 
            "message": "Model not loaded. Check Backend Console (Black Window) for errors."
        }

    image_bytes = await file.read()
    try:
        mask = engine.predict(image_bytes)
        return {"status": "success", "mask": mask.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)