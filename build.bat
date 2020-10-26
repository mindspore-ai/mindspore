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

find "const int ms_version_major =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%a in ('findstr "const int ms_version_major = " version.txt') do (set x=%%a)
set VERSION_MAJOR=%x:~1,1%
find "const int ms_version_minor =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%b in ('findstr "const int ms_versio/retestn_minor = " version.txt') do (set y=%%b)
set VERSION_MINOR=%y:~1,1%
find "const int ms_version_revision =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%c in ('findstr "const int ms_version_revision = " version.txt') do (set z=%%c)
set VERSION_REVISION=%z:~1,1%
del version.txt

echo "======Start building MindSpore Lite %VERSION_MAJOR%.%VERSION_MINOR%.%VERSION_REVISION%======"

SET threads=6
IF NOT "%2%" == "" (
    SET threads=%2%
)

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build
IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)
cd %BUILD_PATH%
IF NOT EXIST "%BUILD_PATH%/mindspore" (
    md "mindspore"
)

cd %BUILD_PATH%/mindspore
IF "%1%" == "lite" (
    cmake -DPLATFORM_ARM64=off -DSUPPORT_TRAIN=off ^
    -DENABLE_TOOLS=on -DENABLE_CONVERTER=on -DBUILD_TESTCASES=off ^
    -DCMAKE_BUILD_TYPE=Release -DSUPPORT_GPU=off -DBUILD_MINDDATA=off -DOFFLINE_COMPILE=off ^
    -DMS_VERSION_MAJOR=%VERSION_MAJOR% -DMS_VERSION_MINOR=%VERSION_MINOR% -DMS_VERSION_REVISION=%VERSION_REVISION% ^
    -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/mindspore/lite"
) ELSE (
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON ^
    -G "CodeBlocks - MinGW Makefiles" ../..
)
IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :run_fail
)

cmake --build . --target package -- -j%threads%
IF NOT %errorlevel% == 0 (
    echo "build fail."
    call :run_fail
)

IF EXIST "%BASE_PATH%/output" (
    cd %BASE_PATH%/output
    rd /s /q _CPack_Packages
)

goto run_eof

:run_fail
    cd %BASE_PATH%
    set errorlevel=1
    EXIT /b %errorlevel%

:run_eof
    cd %BASE_PATH%