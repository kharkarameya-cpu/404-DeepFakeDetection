@echo off
REM Build the standalone Windows GUI app for the DeepFakeDetector test harness.
REM
REM   --onedir is used (not --onefile) so the app starts instantly: a onefile
REM   exe packs ~340 MB into itself and re-extracts that archive on *every* launch
REM   (20-30s of waiting). The onedir folder only unpacks once and then runs
REM   immediately thereafter.
REM
REM Output: dist\DeepFakeDetectorTester\DeepFakeDetectorTester.exe
REM Requirements:
REM   pip install -r requirements.txt -r requirements-gui.txt
setlocal
cd /d "%~dp0.."
python -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name DeepFakeDetectorTester ^
  --paths "." ^
  --collect-all mediapipe ^
  --collect-data imageio_ffmpeg ^
  --hidden-import h5py ^
  --add-data "src\preprocessing\models;models" ^
  "gui\app.py"
echo EXIT=%ERRORLEVEL%
