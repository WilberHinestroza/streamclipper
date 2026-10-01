@echo off
cd /d "%~dp0"
python -m streamclipper.gui
if %errorlevel% neq 0 (
    echo.
    echo Hubo un error al abrir la interfaz. Revisa que Python y las dependencias
    echo esten instalados ^(ver README.md^): pip install -r requirements.txt
    pause
)
