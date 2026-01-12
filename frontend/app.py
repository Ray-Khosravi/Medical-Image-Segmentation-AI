import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2

# تنظیمات صفحه
st.set_page_config(page_title="Medical AI Segmentor", layout="wide", page_icon="🧬")
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff;}
    div.stButton > button:first-child {background-color: #2ecc71; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medical Image Segmentation")

API_URL = "http://localhost:8000/segment"

# --- تابع ترکیب تصاویر (دقیقاً مشابه منطق نوت‌بوک) ---
def apply_mask_overlay(original_pil, mask_np, alpha=0.4):
    """
    original_pil: تصویر اصلی (PIL)
    mask_np: خروجی مدل با شیپ (3, H, W) -> طبق استاندارد PyTorch
    """
    try:
        # 1. تبدیل تصویر اصلی به آرایه NumPy و فرمت RGB
        # (Height, Width, 3)
        background = np.array(original_pil.convert("RGB"))
        bg_h, bg_w = background.shape[:2]

        # 2. اصلاح ابعاد ماسک (نکته کلیدی که در نوت‌بوک با permute انجام می‌شد)
        # ورودی مدل (3, 224, 224) است. ما باید آن را به (224, 224, 3) تبدیل کنیم.
        if mask_np.shape[0] == 3: 
            # تبدیل (Channel, Height, Width) -> (Height, Width, Channel)
            mask_np = np.transpose(mask_np, (1, 2, 0))
        
        # 3. تغییر سایز ماسک به اندازه تصویر اصلی
        # چون مدل روی 224x224 کار کرده ولی تصویر اصلی بزرگتر است (مثلاً 266x266)
        # باید ماسک را Resize کنیم تا فیتِ عکس اصلی شود.
        mask_resized = cv2.resize(mask_np, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)

        # 4. آماده‌سازی ماسک رنگی برای نمایش
        # در نوت‌بوک: کانال 0=قرمز، 1=سبز، 2=آبی
        # مقادیر خروجی مدل احتمالا بین 0 و 1 (Sigmoid) هستند.
        
        # یک لایه رنگی خالی میسازیم
        colored_mask = np.zeros_like(background)
        
        # آستانه گذاری (Threshold) - مشابه نوت‌بوک که Dice Score میگیرد
        threshold = 0.5
        
        # کانال قرمز (Large Bowel)
        colored_mask[:, :, 0] = np.where(mask_resized[:, :, 0] > threshold, 255, 0)
        # کانال سبز (Small Bowel)
        colored_mask[:, :, 1] = np.where(mask_resized[:, :, 1] > threshold, 255, 0)
        # کانال آبی (Stomach)
        colored_mask[:, :, 2] = np.where(mask_resized[:, :, 2] > threshold, 255, 0)

        # 5. ترکیب (Overlay)
        # فقط جاهایی که ماسک وجود دارد را با شفافیت ترکیب میکنیم
        mask_indices = np.any(colored_mask > 0, axis=-1)
        
        overlay = background.copy()
        if np.any(mask_indices):
            overlay[mask_indices] = cv2.addWeighted(
                background[mask_indices], 1 - alpha, 
                colored_mask[mask_indices], alpha, 
                0
            )
            
        return overlay

    except Exception as e:
        st.error(f"Error in overlay logic: {e}")
        return np.array(original_pil)


# --- تابع کمکی پردازش نمایش ---
def process_image_for_display(image):
    if image is None: return None
    # هندل کردن تصاویر 16 بیتی پزشکی
    if image.mode in ['I;16', 'I']:
        img_array = np.array(image)
        # نرمال‌سازی مینیمم-ماکزیمم (مثل نوت‌بوک)
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        else:
            img_array = np.zeros_like(img_array)
        return Image.fromarray(img_array.astype(np.uint8))
    return image.convert("RGB")

# --- رابط کاربری ---
uploaded_files = st.file_uploader("Upload Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    
    st.sidebar.header(f"📂 Files: {len(uploaded_files)}")
    slice_index = st.sidebar.slider("Select Slice", 0, len(uploaded_files)-1, 0)
    
    current_file = uploaded_files[slice_index]
    
    st.subheader(f"Analyzing: {current_file.name}")
    
    # لود عکس
    raw_image = Image.open(current_file)
    display_image = process_image_for_display(raw_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image, caption="Original Scan", use_container_width=True)

    if st.sidebar.button("🔍 Run Model", type="primary"):
        with st.spinner("Segmenting..."):
            try:
                # ارسال به API
                current_file.seek(0)
                mime = "image/png" if current_file.name.endswith("png") else "image/jpeg"
                files = {"file": (current_file.name, current_file.getvalue(), mime)}
                
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # دریافت ماسک خام از API
                        raw_mask = np.array(result["mask"])
                        
                        # --- اعمال Overlay با تابع اصلاح شده ---
                        final_result = apply_mask_overlay(display_image, raw_mask, alpha=0.5)
                        
                        with col2:
                            st.image(final_result, caption="AI Prediction", use_container_width=True)
                            st.markdown("""
                            **Classes:**
                            <span style='color:#ff4b4b'>■ Large Bowel</span> &nbsp;
                            <span style='color:#2ecc71'>■ Small Bowel</span> &nbsp;
                            <span style='color:#4b8bf5'>■ Stomach</span>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"AI Error: {result.get('message')}")
                else:
                    st.error(f"Server Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")