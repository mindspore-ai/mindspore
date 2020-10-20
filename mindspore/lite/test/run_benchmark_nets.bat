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
@title run_benchmark_nets

SET BASEPATH=%CD%
SET TOOL_PATH=%1
SET TOOL_PATH=%TOOL_PATH:"=%/windows_x64
SET MODEL_PATH=%2
SET MODEL_PATH=%MODEL_PATH:"=%/models/hiai
SET BENCHMARK_BASE=%BASEPATH:"=%/output/benchmark

cd /d %BASEPATH%

IF EXIST "%BASEPATH%/output" (
    rd /s /q output
)
md output

SET OUTPUT_PATH=%BASEPATH%/output

cd /d %OUTPUT_PATH%

IF EXIST benchmark (
    rd /s /q benchmark
)
md benchmark
SET ret_code=0
7z x -r "%TOOL_PATH%/mindspore-lite-*-converter-win-cpu.zip" -o"%BENCHMARK_BASE%"
IF NOT %errorlevel% == 0 (
    echo "Decompression of converter tool fail!"
    SET ret_code=1
)
IF %ret_code% == 0 (
    7z x -r "%TOOL_PATH%/mindspore-lite-*-win-runtime-x86-cpu.zip" -o"%BENCHMARK_BASE%"
    IF NOT %errorlevel% == 0 (
        echo "Decompression of runtime tool fail!"
        SET ret_code=1
    )
)
cd benchmark
md ms
cd mindspore-lite-*-converter-win-cpu/converter
IF %ret_code% == 0 (
    converter_lite --outputFile="%BENCHMARK_BASE%/ms/scan_hms_angle1" --modelFile="%MODEL_PATH%/scan_hms_angle1.tflite" --fmk=TFLITE
    IF NOT %errorlevel% == 0 (
        echo "Model conversion of scan_hms_angle1.tflite fail!"
        SET ret_code=1
    )
)

IF %ret_code% == 0 (
    converter_lite --outputFile="%BENCHMARK_BASE%/ms/scan_hms_detect" --modelFile="%MODEL_PATH%/scan_hms_detect.tflite" --fmk=TFLITE
    IF NOT %errorlevel% == 0 (
        echo "Model conversion of scan_hms_detect.tflite fail!"
        SET ret_code=1
    )
)

cd /d %BENCHMARK_BASE%/mindspore-lite-*-win-runtime-x86-cpu/benchmark
SET INPUT_BASE=%MODEL_PATH%/input_output/input
SET OUTPUT_BASE=%MODEL_PATH%/input_output/output
IF %ret_code% == 0 (
    benchmark --modelFile="%BENCHMARK_BASE%/ms/scan_hms_angle1.ms" --inDataFile="%INPUT_BASE%/scan_hms_angle1.tflite.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/scan_hms_angle1.tflite.ms.out"
    IF NOT %errorlevel% == 0 (
        echo "benchmark scan_hms_angle1 fail!"
        SET ret_code=1
    )
)

IF %ret_code% == 0 (
    benchmark --modelFile="%BENCHMARK_BASE%/ms/scan_hms_detect.ms" --inDataFile="%INPUT_BASE%/scan_hms_detect.tflite.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/scan_hms_detect.tflite.ms.out"
    IF NOT %errorlevel% == 0 (
        echo "benchmark scan_hms_detect fail!"
        SET ret_code=1
    )
)

cd /d %BASEPATH%

IF %ret_code% == 1 (
    SET errorlevel=1
)
