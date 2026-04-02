@echo off
cd /d C:\Users\Gianluca\ai-risk-system-v3

echo === RUN LIVE ===
call venv\Scripts\python.exe -m src.app.run_live

echo === BACKTEST ===
call venv\Scripts\python.exe -m src.portfolio.multi_asset_backtest

echo === DONE ===