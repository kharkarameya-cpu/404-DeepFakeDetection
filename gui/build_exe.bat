@echo off
REM Build the standalone Windows .exe for the test GUI.
REM Produces: dist\DeepFakeDetectorTester.exe
REM Requirements:
REM   pip install -r requirements.txt -r requirements-gui.txt
setlocal
cd /d "%~dp0.."
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name DeepFakeDetectorTester ^
  --paths "." ^
  --collect-all mediapipe ^
  --hidden-import h5py ^
  --add-data "src\preprocessing\models;models" ^
  "gui\app.py"
echo EXIT=%ERRORLEVEL%
