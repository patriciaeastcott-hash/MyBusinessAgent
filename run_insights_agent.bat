@echo off
REM --- 1. Change to the Directory of the Virtual Environment ---
cd "C:\Users\trish\OneDrive\DIGITALABCS\Website\scripts\MyBusinessAgent"

REM --- 2. Activate the Virtual Environment ---
REM The path below should point to the 'activate' script inside your venv's Scripts folder
call "venv\Scripts\activate.bat"

REM --- 3. Run the Python Script ---
REM After activation, you can just use 'python' and the script name
python "digitalabcs_insights_agent.py"

REM --- Optional: Deactivate the Virtual Environment (Good Practice) ---
deactivate

REM --- Optional: Pause to see any errors if running manually ---
REM pause