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
    echo "cmake fail."
    goto run_fail
    )
 
IF "%1%" == "" (
    cmake --build . --target package -- -j6
    ) ELSE (
        cmake --build . --target package -- -j%1%
    )
IF NOT %errorlevel% == 0 (
    echo "build fail."
    goto run_fail
    )

cd %BASEPATH%

goto run_eof

:run_fail
    cd %BASEPATH%
    set errorlevel=1

:run_eof
