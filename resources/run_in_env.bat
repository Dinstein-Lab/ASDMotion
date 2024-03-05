@echo off
REM Activate the conda environment
CALL C:\Users\owner\anaconda3\Scripts\activate.bat open-mmlab

REM Execute the command passed to this batch file
%*

REM Deactivate the environment
CALL conda deactivate