@echo off
@title mindspore_build
 
SET BASEPATH=%CD%
IF NOT EXIST %BASEPATH%/build (
         md "build"
         )
 
cd %BASEPATH%/build
SET BUILD_PATH=%CD%
 
IF NOT EXIST %BUILD_PATH%/mindspore (
         md "mindspore"
         )
 
cd %CD%/mindspore
 
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -G "CodeBlocks - MinGW Makefiles" ../..
IF NOT %errorlevel% == 0 (
    goto run_fail
    )
 
cmake --build . --target all -- -j6
IF NOT %errorlevel% == 0 (
    goto run_fail
    )

cd %BASEPATH%

goto run_eof

:run_fail
    cd %BASEPATH%
    echo "build fail."

:run_eof
