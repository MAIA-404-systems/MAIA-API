@echo off
echo Lancement du serveur API...
python server.py
if %errorlevel% neq 0 (
    echo Erreur lors du lancement du serveur.
    pause
)
pause
