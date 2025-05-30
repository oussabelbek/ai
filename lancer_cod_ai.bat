@echo off
REM Demande les droits administrateur si nécessaire
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo Demande des privilèges d'administrateur...
    goto UACPrompt
) else ( goto gotAdmin )
:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    set params = %*:"=""
    echo UAC.ShellExecute "cmd.exe", "/c %~s0 %params%", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B
:gotAdmin
    rem Si on est ici, on a les droits admin
    pushd "%~dp0"

REM Naviguer vers le dossier du script (si le .bat est dans le même dossier, c'est déjà fait par pushd)
cd /d "H:\ai cod"  
REM Exécuter le script Python
echo Lancement du script IA pour Call of Duty...
python import_cv2.py

REM Garder la fenêtre ouverte après la fin du script pour voir les messages
pause