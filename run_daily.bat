@echo off
cd /d C:\Users\Gianluca\ai-risk-system-v3

echo === RUN LIVE ===
call venv\Scripts\python.exe run_live_safe.py

echo === BACKTEST ===
call venv\Scripts\python.exe -m src.portfolio.multi_asset_backtest

echo === GIT PUSH ===
git add .
git commit -m "auto update"
git push

echo === DONE ===