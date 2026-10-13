$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m venv .venv
if ($LASTEXITCODE) { throw 'venv creation failed' }
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'dependency installation failed' }
& ./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
if ($LASTEXITCODE) { throw 'locked dependency installation failed' }
& ./.venv/Scripts/python.exe prepare.py
if ($LASTEXITCODE) { throw 'data preparation failed' }
& ./.venv/Scripts/python.exe run.py --device 0 --epochs 30
if ($LASTEXITCODE) { throw 'training failed' }
