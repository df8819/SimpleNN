@echo off
cd /d "%~dp0"

REM Perform a Git pull to update the repository
git pull

REM Install dependencies from requirements.txt using pip
pip install -r requirements.txt

REM Run the Python script
python main.py
