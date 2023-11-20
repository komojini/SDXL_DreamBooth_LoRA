@echo off

SET ENV_DIR=venv

IF NOT EXIST "%ENV_DIR%" (
    echo Creating virtual environment...
    python -m venv %ENV_DIR%
)

echo To activate the virtual environment, run '%ENV_DIR%\Scripts\activate'


