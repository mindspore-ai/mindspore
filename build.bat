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

@echo off
echo Start build at: %date% %time%

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=8
SET ENABLE_GITEE=OFF

set VERSION_STR=''
for /f "tokens=1" %%a in (version.txt) do (set VERSION_STR=%%a)

ECHO %2%|FINDSTR "^[0-9][0-9]*$"
IF %errorlevel% == 0 (
    SET threads=%2%
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
    echo "======Start building MindSpore Lite %VERSION_STR%======"
    rd /s /q "%BASE_PATH%\output"
    (git log -1 | findstr "^commit") > %BUILD_PATH%\.commit_id
    IF defined VisualStudioVersion (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "Ninja" "%BASE_PATH%/mindspore/lite"
    ) ELSE (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/mindspore/lite"
    )
) ELSE (
    IF "%1%" == "ms_vs_gpu" (
        echo "======Start gen VS2019 Project for MS gpu ======"
        cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_GPU=ON -DGPU_BACKEND_CUDA=ON -DMS_REQUIRE_CUDA_VERSION=11.1 -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
            -G "Visual Studio 16 2019" -A x64 ../..
    ) ELSE (
        IF "%1%" == "ms_vs_cpu" (
            echo "======Start gen VS2019 Project for MS cpu ======"
            cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G "Visual Studio 16 2019" -A x64 ../..
        ) ELSE (
            echo "======Start gen MinGW64 Project for MS cpu ======"
            cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G "CodeBlocks - MinGW Makefiles" ../..
        )
    )
)
IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :clean
    EXIT /b 1
)

IF "%1%" == "ms_vs_gpu" (
    cmake --build . --config Release --target package
) ELSE (
    IF "%1%" == "ms_vs_cpu" (
        cmake --build . --config Release --target package
    ) ELSE (
        cmake --build . --target package -- -j%threads%
    )
)

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
        if EXIST "%BASE_PATH%/output/_CPack_Packages" (
             rd /s /q _CPack_Packages
        )
    )
    cd %BASE_PATH%

@echo off
echo End build at: %date% %time%