@echo off
TITLE Medical AI Automation
echo ===================================================
echo    Medical Segmentation Auto-Launcher
echo ===================================================

:: --- مرحله ۱: بررسی و ساخت محیط مجازی ---
if not exist venv (
    echo [1/4] Virtual Environment not found. Creating 'venv'...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Error: Python is not installed or not in PATH.
        pause
        exit
    )
    echo ✅ Virtual Environment created.
) else (
    echo [1/4] Virtual Environment found. Skipping creation.
)

:: --- مرحله ۲: فعال‌سازی محیط ---
call venv\Scripts\activate

:: --- مرحله ۳: نصب کتابخانه‌ها ---
echo [2/4] Installing/Updating Dependencies...
echo       (This might take a while for the first time)
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ❌ Error installing requirements. Check your internet connection.
    pause
    exit
)
echo ✅ Dependencies are ready.

:: --- مرحله ۴: اجرای برنامه ---
echo [3/4] Launching Backend API...
start "Backend Server" cmd /k "call venv\Scripts\activate && cd backend && python main.py"

echo [4/4] Waiting for server to start...
timeout /t 6 /nobreak >nul

echo 🚀 Launching Frontend...
start "Frontend UI" cmd /k "call venv\Scripts\activate && cd frontend && streamlit run app.py"

echo.
echo ===================================================
echo    SUCCESS! System is fully operational.
echo    Backend: http://localhost:8000
echo    Frontend: http://localhost:8501
echo ===================================================
echo.
pause