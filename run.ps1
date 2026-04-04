# Quick local runner for the final Flask app

& .\.venv\Scripts\Activate.ps1

d:/Downloads/files/.venv/Scripts/python.exe -m pip install -r requirements.txt
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
