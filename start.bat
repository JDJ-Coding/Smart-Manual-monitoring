@echo off
chcp 65001 >nul
echo ========================================
echo Smart Manual Assistant
echo ========================================
echo.

REM Check if MY_API_KEY_GOOGLE is set
if "%MY_API_KEY_GOOGLE%"=="" (
    echo [WARNING] MY_API_KEY_GOOGLE 환경변수가 설정되지 않았습니다.
    echo Gemini API 키를 설정하려면:
    echo   set MY_API_KEY_GOOGLE=your_api_key
    echo.
)

REM Check if manual_db exists
if not exist "manual_db\index.pkl" (
    echo [INFO] 벡터 DB가 없습니다. 먼저 build_db.py를 실행하세요.
    echo.
    set /p BUILD="지금 벡터 DB를 구축하시겠습니까? (y/n): "
    if /i "!BUILD!"=="y" (
        echo.
        echo [INFO] 벡터 DB 구축 중...
        python build_db.py
        if errorlevel 1 (
            echo [ERROR] 벡터 DB 구축 실패
            pause
            exit /b 1
        )
        echo [SUCCESS] 벡터 DB 구축 완료!
        echo.
    ) else (
        echo [INFO] 벡터 DB 없이 앱을 시작합니다.
        echo.
    )
)

echo [INFO] Streamlit 앱을 시작합니다...
echo [INFO] 브라우저에서 http://localhost:8501 을 열어주세요.
echo [INFO] 종료하려면 Ctrl+C를 누르세요.
echo.

streamlit run app.py

pause
