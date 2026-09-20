@echo off
setlocal
python --version
python -m pip install -r requirements.txt
python pedestrian_behavior_simulation_corrected.py
pause
