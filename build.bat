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

SET BASEPATH=%CD%
SET BUILD_PATH=%BASEPATH%/build

find "const int ms_version_major =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%a in ('findstr "const int ms_version_major = " version.txt') do (set x=%%a)
set VERSION_MAJOR=%x:~1,1%
find "const int ms_version_minor =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%b in ('findstr "const int ms_version_minor = " version.txt') do (set y=%%b)
set VERSION_MINOR=%y:~1,1%
find "const int ms_version_revision =" mindspore\lite\include\version.h > version.txt
for /f "delims=\= tokens=2" %%c in ('findstr "const int ms_version_revision = " version.txt') do (set z=%%c)
set VERSION_REVISION=%z:~1,1%
del version.txt
echo "======Start building MindSpore Lite %VERSION_MAJOR%.%VERSION_MINOR%.%VERSION_REVISION%======"

IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)

cd %BUILD_PATH%

IF NOT EXIST "%BUILD_PATH%/mindspore" (
    md "mindspore"
)

cd %CD%/mindspore

IF "%1%" == "lite" (
    call :run_cmake
    IF errorlevel 1 (
        echo "cmake fail one time."
        call :gene_protobuf
        call :gene_flatbuffer
        call :run_cmake
        IF errorlevel 1 (
            echo "cmake fail."
            call :run_fail
        )
    ) ELSE (
        call :gene_protobuf
        call :gene_flatbuffer
    )

    cd %BUILD_PATH%/mindspore
    IF "%2%" == "" (
        cmake --build . --target package -- -j6
    ) ELSE (
        cmake --build . --target package -- -j%2%
    )
    IF errorlevel 1 (
        echo "build fail."
        call :run_fail
    ) ELSE (
        cd %BASEPATH%/output
        rd /s /q _CPack_Packages
    )
) ELSE (
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON ^
    -G "CodeBlocks - MinGW Makefiles" ../..
    IF NOT %errorlevel% == 0 (
        echo "cmake fail."
        call :run_fail
    )

    IF "%1%" == "" (
        cmake --build . --target package -- -j6
    ) ELSE (
        cmake --build . --target package -- -j%1%
    )
    IF NOT %errorlevel% == 0 (
        echo "build fail."
        call :run_fail
    )
)

cd %BASEPATH%

goto run_eof

:run_cmake
    cd %BUILD_PATH%/mindspore
    cmake -DBUILD_DEVICE=on -DBUILD_CONVERTER=on -DPLATFORM_ARM64=off -DSUPPORT_TRAIN=off ^
    -DCMAKE_BUILD_TYPE=Release -DSUPPORT_GPU=off -DBUILD_MINDDATA=off -DOFFLINE_COMPILE=off ^
    -DMS_VERSION_MAJOR=%VERSION_MAJOR% -DMS_VERSION_MINOR=%VERSION_MINOR% -DMS_VERSION_REVISION=%VERSION_REVISION% ^
    -G "CodeBlocks - MinGW Makefiles" "%BASEPATH%/mindspore/lite"
GOTO:EOF

:gene_protobuf
    IF NOT DEFINED MSLIBS_CACHE_PATH (
        cd /d %BASEPATH%/build/mindspore/_deps/protobuf-src/_build
    ) ELSE (
        cd /d %MSLIBS_CACHE_PATH%/protobuf_*/bin
    )

    SET PROTO_SRC_DIR=%BASEPATH%/mindspore/lite/tools/converter/parser/caffe
    protoc "%PROTO_SRC_DIR%/*.proto" --proto_path="%PROTO_SRC_DIR%" --cpp_out="%PROTO_SRC_DIR%"

    SET PROTO_SRC_DIR=%BASEPATH%/mindspore/lite/tools/converter/parser/onnx
    protoc "%PROTO_SRC_DIR%/*.proto" --proto_path="%PROTO_SRC_DIR%" --cpp_out="%PROTO_SRC_DIR%"
    cd /d %BUILD_PATH%/mindspore
GOTO:EOF

:gene_flatbuffer
    IF NOT DEFINED MSLIBS_CACHE_PATH (
        cd /d %BASEPATH%/build/mindspore/_deps/flatbuffers-src/_build
    ) ELSE (
        cd /d %MSLIBS_CACHE_PATH%/flatbuffers_*/bin
    )

    SET FLAT_DIR=%BASEPATH%/mindspore/lite/schema
    flatc -c -b -o "%FLAT_DIR%" "%FLAT_DIR%/*.fbs"
    flatc -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "%FLAT_DIR%/inner" "%FLAT_DIR%/*.fbs"

    SET FLAT_DIR=%BASEPATH%/mindspore/lite/tools/converter/parser/tflite
    flatc -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "%FLAT_DIR%" "%FLAT_DIR%/*.fbs"
    cd /d %BUILD_PATH%/mindspore
GOTO:EOF

:run_fail
    cd %BASEPATH%
    set errorlevel=1
    exit /b %errorlevel%

:run_eof
