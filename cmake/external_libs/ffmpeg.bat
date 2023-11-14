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

echo Start complie FFmpeg software

SET MINGW64_NAME="mingw64.exe"
SET PROCESS_NAME=mintty.exe
SET TEMP_FILE="temp.txt"

del %TEMP_FILE%
where %MINGW64_NAME% > %TEMP_FILE%
SET /p file_path=<%TEMP_FILE%
del %TEMP_FILE%

SET MSYS64_PATH=%file_path:~0,-12%
SET FFMPEG_DLL_SOURCE_PATH=%MSYS64_PATH%\home\ffmpeg\install_ffmpeg_lib\bin
SET FFMPEG_DLL_PATH=%BASE_PATH%\build\mindspore\ffmpeg_lib
SET FFMPEG_GITHUB_DOWNLOAD_PATH=https://ffmpeg.org/releases/ffmpeg-5.1.2.tar.gz
SET FFMPEG_DOWNLOAD_PATH=https://tools.mindspore.cn/libs/ffmpeg/ffmpeg-5.1.2.tar.gz

pushd %MSYS64_PATH%

start cmd /c "msys2_shell.cmd -mingw64 -no-start -c 'cd /home; rm -rf ffmpeg; mkdir ffmpeg; cd /home/ffmpeg; wget %FFMPEG_DOWNLOAD_PATH%; tar -xzvf ffmpeg-5.1.2.tar.gz; mkdir build; mkdir install_ffmpeg_lib; cd build; ../ffmpeg-5.1.2/configure --prefix=/home/ffmpeg/install_ffmpeg_lib --disable-programs --disable-doc --disable-postproc --disable-libxcb --disable-hwaccels --disable-static --enable-shared --disable-decoder=av1 --toolchain=msvc; make -j4; make install'"

ping 127.0.0.1 -n 10 >nul

:LOOP
tasklist | findstr /i "%PROCESS_NAME%"
if errorlevel 1 (
    echo The process %PROCESS_NAME% is not running.
    goto :END_LOOP
) else (
    echo The process %PROCESS_NAME% is running.
)
ping 127.0.0.1 -n 10 >nul
goto LOOP

:END_LOOP
echo "FFmpeg is ended. Continuing the next processing."

xcopy %FFMPEG_DLL_SOURCE_PATH% %FFMPEG_DLL_PATH% /E /I /Y

popd