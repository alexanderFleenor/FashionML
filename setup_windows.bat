@echo off
echo ==========================================
echo   Fashion Outfit Recommendation Setup
echo ==========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch (CPU version for simplicity)
echo.
echo Installing PyTorch (this may take a few minutes)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: Install other requirements
echo.
echo Installing other dependencies...
pip install -r requirements.txt

:: Verify installation
echo.
echo ==========================================
echo   Verifying installation...
echo ==========================================
python -c "import torch; import cv2; import sklearn; print('All packages installed successfully!')"
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly.
) else (
    echo.
    echo SUCCESS! Setup complete.
)

echo.
echo ==========================================
echo   To run the notebooks:
echo ==========================================
echo   1. Open a command prompt in this folder
echo   2. Run: venv\Scripts\activate
echo   3. Run: jupyter notebook
echo   4. Open notebooks/05_full_pipeline_demo.ipynb
echo ==========================================
echo.
pause
