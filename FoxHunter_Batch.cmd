@echo off
setlocal
set "batchgui=%~dp0scripts\foxhunter_batch_gui.py"
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" goto bundled
py -3 "%batchgui%"
if errorlevel 1 pause
exit /b
:bundled
"%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "%batchgui%"
if errorlevel 1 pause
