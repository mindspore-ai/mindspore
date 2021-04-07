@rem Copyright 2020 Huawei Technologies Co., Ltd
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
@title mindspore_build

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=6
SET X86_64_SIMD=off
SET ENABLE_GITEE=OFF

set VERSION_MAJOR=''
set VERSION_MINOR=''
set ERSION_REVISION=''

for /f "delims=\= tokens=2" %%a in ('findstr /C:"const int ms_version_major = " mindspore\lite\include\version.h') do (set x=%%a)
set VERSION_MAJOR=%x:~1,1%
for /f "delims=\= tokens=2" %%b in ('findstr /C:"const int ms_version_minor = " mindspore\lite\include\version.h') do (set y=%%b)
set VERSION_MINOR=%y:~1,1%
for /f "delims=\= tokens=2" %%c in ('findstr /C:"const int ms_version_revision = " mindspore\lite\include\version.h') do (set z=%%c)
set VERSION_REVISION=%z:~1,1%

echo "======Start building MindSpore Lite %VERSION_MAJOR%.%VERSION_MINOR%.%VERSION_REVISION%======"

ECHO %2%|FINDSTR "^[0-9][0-9]*$"
IF %errorlevel% == 0 (
    SET threads=%2%
) ELSE (
    IF NOT "%2%" == "" (
        IF "%2%" == "avx" (
            SET X86_64_SIMD=avx
        ) ELSE IF "%2%" == "sse" (
            SET X86_64_SIMD=sse
        ) ELSE IF "%2%" == "off" (
            SET X86_64_SIMD=off
        ) ELSE IF "%2%" == "avx512" (
            SET X86_64_SIMD=avx512
        ) ELSE (
            echo "MindSpore_lite the second parameter must in [avx, avx512, sse, off], but now is [%2%]"
            call :clean
            EXIT /b 1
        )
        IF NOT "%3%" == "" (
            SET threads=%3%
        )
    )
)

IF "%FROM_GITEE%" == "1" (
    echo "DownLoad from gitee"
    SET ENABLE_GITEE=ON
)

IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)
cd %BUILD_PATH%
IF NOT EXIST "%BUILD_PATH%/mindspore" (
    md "mindspore"
)

cd %BUILD_PATH%/mindspore
IF "%1%" == "lite" (
    cmake --build "%BUILD_PATH%\mindspore" --target clean
    rd /s /q "%BASE_PATH%\output"
    (git log -1 | findstr "^commit") > %BUILD_PATH%\.commit_id
    cmake -DPLATFORM_ARM64=off -DSUPPORT_TRAIN=off ^
    -DENABLE_TOOLS=on -DENABLE_CONVERTER=on -DBUILD_TESTCASES=off ^
    -DCMAKE_BUILD_TYPE=Release -DSUPPORT_GPU=off -DBUILD_MINDDATA=off -DOFFLINE_COMPILE=off ^
    -DMS_VERSION_MAJOR=%VERSION_MAJOR% -DMS_VERSION_MINOR=%VERSION_MINOR% -DMS_VERSION_REVISION=%VERSION_REVISION% ^
    -DX86_64_SIMD=%X86_64_SIMD% ^
    -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/mindspore/lite"
) ELSE (
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
    -G "CodeBlocks - MinGW Makefiles" ../..
)
IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :clean
    EXIT /b 1
)

cmake --build . --target package -- -j%threads%
IF NOT %errorlevel% == 0 (
    echo "build fail."
    call :clean
    EXIT /b 1
)

call :clean
EXIT /b 0

:clean
    IF EXIST "%BASE_PATH%/output" (
        cd %BASE_PATH%/output
        rd /s /q _CPack_Packages
    )
    cd %BASE_PATH%