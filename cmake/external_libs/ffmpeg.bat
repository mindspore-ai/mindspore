@rem Copyright 2024 Huawei Technologies Co., Ltd
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem ============================================================================
@echo off

echo Start compile FFmpeg software at: %date% %time%

SET MINGW64_NAME="mingw64.exe"
SET PROCESS_NAME=mintty.exe
SET TEMP_FILE="temp.txt"

where %MINGW64_NAME% > %TEMP_FILE%
if errorlevel 1 (
    echo The software 'MSYS2' is not installed, please refer to the link https://www.mindspore.cn/install to install it first.
    EXIT /b 1
)
SET /p file_path=<%TEMP_FILE%
del %TEMP_FILE%

echo Checking for running %PROCESS_NAME% process...
tasklist /FI "IMAGENAME eq %PROCESS_NAME%" 2>NUL | find /I "%PROCESS_NAME%" >NUL
if %errorlevel%==0 (
    echo Found running %PROCESS_NAME% process. Attempting to terminate...
    taskkill /F /IM %PROCESS_NAME% >NUL 2>&1
    if %errorlevel%==0 (
        echo %PROCESS_NAME% process terminated successfully.
    ) else (
        echo Warning: Failed to terminate %PROCESS_NAME% process.
    )
) else (
    echo No running %PROCESS_NAME% process found.
)

SET MSYS64_PATH=%file_path:~0,-12%
SET FFMPEG_DLL_SOURCE_PATH=%MSYS64_PATH%\home\ffmpeg\install_ffmpeg_lib\bin
SET FFMPEG_COMPILE_SOURCE_PATH=%MSYS64_PATH%\home\ffmpeg\install_ffmpeg_lib
SET FFMPEG_DLL_PATH=%BASE_PATH%\build\mindspore\ffmpeg_lib
SET MINDSPORE_FFMPEG_PATCH_PATH=%BASE_PATH%\third_party\patch\ffmpeg
SET MSYS_FFMPEG_PATCH_PATH=%MSYS64_PATH%\home\patch
SET FFMPEG_ERROR_LOG=%MSYS64_PATH%\home\ffmpeg\build\ffbuild\config.log


SET SOFTWARE_NAME=FFmpeg
SET SOFTWARE_VERSION=5.1.4
SET FFMPEG_GITHUB_DOWNLOAD_PATH=https://ffmpeg.org/releases/ffmpeg-%SOFTWARE_VERSION%.tar.gz
SET FFMPEG_MINDSPORE_DOWNLOAD_PATH=https://tools.mindspore.cn/libs/ffmpeg/ffmpeg-%SOFTWARE_VERSION%.tar.gz
SET FFMPEG_DOWNLOAD_PATH=%FFMPEG_MINDSPORE_DOWNLOAD_PATH%
SET FFMPEG_COMPILATION_OPTIONS=--disable-programs --disable-doc --disable-postproc --disable-libxcb --disable-hwaccels --disable-static --enable-shared --disable-decoder=av1 --toolchain=msvc
SET FFMPEG_INFO_CONTEXT=%SOFTWARE_NAME%-%SOFTWARE_VERSION%-%FFMPEG_GITHUB_DOWNLOAD_PATH%-%FFMPEG_MINDSPORE_DOWNLOAD_PATH%-%FFMPEG_COMPILATION_OPTIONS%
SET FFMPEG_LIB_NAME=avcodec-avdevice-avfilter-avformat-avutil-swresample-swscale


IF "%ENABLE_FFMPEG_DOWNLOAD%"=="ON" (
    SET FFMPEG_DOWNLOAD_PATH=%FFMPEG_MINDSPORE_DOWNLOAD_PATH%
) ELSE (
    SET FFMPEG_DOWNLOAD_PATH=%FFMPEG_GITHUB_DOWNLOAD_PATH%
)

rem move Mindspore patch to msys directory.
if exist %MSYS_FFMPEG_PATCH_PATH% (
    rd /s /q %MSYS_FFMPEG_PATCH_PATH%
)
xcopy %MINDSPORE_FFMPEG_PATCH_PATH%\*.* %MSYS_FFMPEG_PATCH_PATH% /E /I /Y

pushd %MSYS64_PATH%

rem calculate cve hash
SET CVE_PATH=%MSYS_FFMPEG_PATCH_PATH%
SET CVE_HASH_FILE="cve_hash.txt"
SET CVE_HASH_TEMP_FILE="cve_hash_temp.txt"

(for /r "%CVE_PATH%" %%f in (*) do (
    certutil -hashfile "%%f" SHA256
)) > "%CVE_HASH_FILE%"

SET CVE_OUT_HASH_VALUE=
rem call :CalculateHash %CVE_HASH_FILE% %CVE_HASH_TEMP_FILE% CVE_OUT_HASH_VALUE
certutil -hashfile %CVE_HASH_FILE% SHA256 > %CVE_HASH_TEMP_FILE%
for /f "usebackq skip=1 delims=" %%i in (%CVE_HASH_TEMP_FILE% ) do (
        SET CVE_OUT_HASH_VALUE=%%i
        goto :endloop
    )
:endloop
echo Finish calculate CVE sha256
echo CVE OUT HASH VALUE: %CVE_OUT_HASH_VALUE%

SET FFMPEG_SHA256_CONTEXT=%FFMPEG_INFO_CONTEXT%-%CVE_OUT_HASH_VALUE%
echo FFMPEG_SHA256_CONTEXT: %FFMPEG_SHA256_CONTEXT%

rem calculate the hash value of the ffmpeg cache
SET OUT_HASH_VALUE=
SET HASH_FILE="hash_file.txt"
SET HASH_TEMP_FILE="hash_temp.txt"
echo %FFMPEG_SHA256_CONTEXT% > %HASH_FILE%
rem call :CalculateHash %HASH_FILE% %HASH_TEMP_FILE% OUT_HASH_VALUE
certutil -hashfile %HASH_FILE% SHA256 > %HASH_TEMP_FILE%
for /f "usebackq skip=1 delims=" %%i in (%HASH_TEMP_FILE% ) do (
        SET OUT_HASH_VALUE=%%i
        goto :endloop
    )
:endloop
echo Finish calculate FFmpeg sha256
echo FFMPEG OUT HASH VALUE: %OUT_HASH_VALUE%

SET FFMPEG_CACHE_DIR=%MSYS64_PATH%\home\mslib\%SOFTWARE_NAME%_%SOFTWARE_VERSION%_%OUT_HASH_VALUE%
echo Get cache dir is: %FFMPEG_CACHE_DIR%


rem Check whether the cache directory exists
IF EXIST %FFMPEG_CACHE_DIR% (
    echo The cache file: %FFMPEG_CACHE_DIR% is exist
    call :dll_file_exist
    if errorlevel 1 (
        echo Checking DLL file failed
        rd /s /q %FFMPEG_CACHE_DIR%
        mkdir %FFMPEG_CACHE_DIR%
    ) else (
        echo Checking DLL file success
        xcopy %FFMPEG_CACHE_DIR% %FFMPEG_DLL_PATH% /E /I /Y
        popd
        echo End cache FFmpeg software at: %date% %time%
        EXIT /b 0
    )
) ELSE (
    echo The cache file is not exist, need to recompile.
    mkdir %FFMPEG_CACHE_DIR%
)

rem Create the patch script
SET MSYS_FFMPEG_PATCH_SHELL=%MSYS64_PATH%\home\shell
if exist %MSYS_FFMPEG_PATCH_SHELL% (
    rd /s /q %MSYS_FFMPEG_PATCH_SHELL%
)
mkdir %MSYS_FFMPEG_PATCH_SHELL%
echo cd /home/ffmpeg/ffmpeg-%SOFTWARE_VERSION% >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh
echo for patch in /home/patch/*.patch >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh
echo do echo "Applying $patch" >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh
echo patch -p1 ^< "$patch" >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh
echo done >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh
echo echo "Complete patching." >> %MSYS_FFMPEG_PATCH_SHELL%\patch_script.sh

rem Execute compilation
start cmd /c "msys2_shell.cmd -mingw64 -no-start -c 'cd /home; rm -rf ffmpeg; mkdir ffmpeg; cd /home/ffmpeg; wget %FFMPEG_DOWNLOAD_PATH%; tar -xzvf ffmpeg-%SOFTWARE_VERSION%.tar.gz; cd /home/shell; bash patch_script.sh; cd /home/ffmpeg; mkdir build; mkdir install_ffmpeg_lib; cd build; ../ffmpeg-%SOFTWARE_VERSION%/configure --prefix=/home/ffmpeg/install_ffmpeg_lib %FFMPEG_COMPILATION_OPTIONS%; make -j4; make install'"


ping 127.0.0.1 -n 10 >nul
echo Start compile FFmpeg. Please wait a moment.
echo ...

:LOOP
tasklist | findstr /i "%PROCESS_NAME%" >nul
if errorlevel 1 (
    echo The process %PROCESS_NAME% is not running.
    goto :END_LOOP
)
ping 127.0.0.1 -n 5 >nul
goto LOOP

:END_LOOP
echo FFmpeg compile is ended. Continuing the next processing



xcopy %FFMPEG_COMPILE_SOURCE_PATH% %FFMPEG_DLL_PATH% /E /I /Y
xcopy %FFMPEG_COMPILE_SOURCE_PATH% %FFMPEG_CACHE_DIR% /E /I /Y

call :dll_file_exist
if errorlevel 1 (
    echo FFmpeg compile failed. Please check the file %FFMPEG_ERROR_LOG% for detailed error messages.
    rd /s /q %FFMPEG_CACHE_DIR%
    EXIT /b 1
) else (
    echo FFmpeg compile success
)

popd

echo End complie FFmpeg software at: %date% %time%
EXIT /b 0


:dll_file_exist
    for %%a in ("%FFMPEG_LIB_NAME:-=" "%") do (
        echo Checking for %%a...
        
        if exist "%FFMPEG_CACHE_DIR%\bin\%%a*.dll" (
            echo The file %%a is found in %FFMPEG_CACHE_DIR%
        ) else (
            echo The file %%a is not found in %FFMPEG_CACHE_DIR%
            EXIT /b 1
        )
    )
    EXIT /b 0
