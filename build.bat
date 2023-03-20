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

setlocal EnableDelayedExpansion

@echo off
echo Start build at: %date% %time%

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=8
SET ENABLE_GITEE=OFF
SET ENABLE_MSVC=OFF
set BUILD_TYPE=Release
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

ECHO %1%|FINDSTR "^ms_vs"
IF %errorlevel% == 0 (
    echo "use msvc compiler"
    SET ENABLE_MSVC=ON
) else (
    echo "use mingw compiler"
)

set USE_SCCACHE=off
where sccache
IF %errorlevel% == 0 (
    set USE_SCCACHE=on
    echo "use sccache to speed up compilation"
) ELSE (
    echo "sccache doesn't exist"
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
        for /f "delims=" %%i in ('powershell.exe -ExecutionPolicy Bypass -Command "Get-ChildItem HKLM:\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall | foreach { Get-ItemProperty $_.PsPath } | where { $_.DisplayName -like '*Visual Studio*' -and $_.InstallLocation.Length -gt 0 } | sort InstallDate -Descending | foreach { Join-Path $_.InstallLocation 'VC\Auxiliary\Build'}"') do (call "%%i\vcvars64.bat")
        IF %USE_SCCACHE% == on (
            cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DENABLE_CPU=ON -DENABLE_GPU=ON -DGPU_BACKEND_CUDA=ON -DMS_REQUIRE_CUDA_VERSION=11.1 -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G Ninja ../..
        ) ELSE (
            cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_GPU=ON -DGPU_BACKEND_CUDA=ON -DMS_REQUIRE_CUDA_VERSION=11.1 -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G Ninja ../..
        )
    ) ELSE IF "%1%" == "ms_vs_cpu" (
        echo "======Start gen VS2019 Project for MS cpu ======"
        for /f "delims=" %%i in ('powershell.exe -ExecutionPolicy Bypass -Command "Get-ChildItem HKLM:\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall | foreach { Get-ItemProperty $_.PsPath } | where { $_.DisplayName -like '*Visual Studio*' -and $_.InstallLocation.Length -gt 0 } | sort InstallDate -Descending | foreach { Join-Path $_.InstallLocation 'VC\Auxiliary\Build'}"') do (call "%%i\vcvars64.bat")
        IF %USE_SCCACHE% == on (
            cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G Ninja ../..
        ) ELSE (
            cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G Ninja ../..
        )
    ) ELSE IF "%1%" == "ms_vs_cpu_debug" (
        echo "======Start gen VS2019 Project for MS cpu debug======"
        cmake -DCMAKE_BUILD_TYPE=Debug -DDEBUG_MODE=ON -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
            -G "Visual Studio 16 2019" -A x64 ../..
        set BUILD_TYPE=Debug
    ) ELSE (
        echo "======Start gen MinGW64 Project for MS cpu ======"
        cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
                -G "CodeBlocks - MinGW Makefiles" ../..
    )
)
IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :clean
    EXIT /b 1
)

IF ON == %ENABLE_MSVC% (
    cmake --build . --config %BUILD_TYPE% --target package
) ELSE (
    cmake --build . --target package -- -j%threads%
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