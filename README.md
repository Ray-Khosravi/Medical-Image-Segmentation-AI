🏥 End-to-End Medical AI: Multi-Class Abdominal Organ Segmentation

🌟 Overview

This project delivers a high-precision medical imaging pipeline designed to segment critical abdominal organs (Stomach, Small Bowel, Large Bowel) from MRI/CT scans. Originally developed for a Kaggle challenge, this system has been re-engineered into a production-ready inference suite with a focus on clinical interpretability and real-time performance.

🎯 The Challenge

Accurate segmentation in medical imaging is a bottleneck for radiotherapy and surgical planning. This project automates the pixel-level identification of organs, reducing the workload for radiologists and minimizing human error in high-stakes environments.

🚀 Key Features

Custom Neural Architecture: Implemented from scratch (no boilerplate libraries) to ensure maximum control over the feature extraction layers.

End-to-End Pipeline: Covers everything from raw DICOM/PNG preprocessing and normalization to real-time mask generation.

Multi-Class Mastery: Simultaneously segments three distinct organ classes with handling for overlapping boundaries and varying contrast.

Interactive Deployment: Includes a Streamlit-based dashboard for interactive slice-by-slice analysis and model validation.


🛠️ Technical Stack & ArchitectureDeep Learning: PyTorch (Model Architecture & Training)


Computer Vision: OpenCV, Albumentations (Advanced Data Augmentation)

Frontend/UI: Streamlit 

Backend: Modular Python Architecture (Clean Code principles)

📊 Results & Visualization

The model performs semantic segmentation on abdominal slices with high Dice coefficients. Below is a snapshot of the AI predicting organ boundaries compared to the original scan:


<img width="1280" height="637" alt="image" src="https://github.com/user-attachments/assets/42e96217-30a1-40d1-a9ab-23ba07666264" />


👋 Contact & Connect

Ray Khosravi - AI / Machine Learning Engineer

[LinkedIn](https://www.linkedin.com/in/raoufkhosravi/)

raykhosravi1993@gmail.com

