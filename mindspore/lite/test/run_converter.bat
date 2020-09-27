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
@title run_converter

SET BASEPATH=%CD%
SET TOOL_PATH=%1
SET TOOL_PATH=%TOOL_PATH:"=%/windows_x64
SET MODEL_PATH=%2
SET MODEL_PATH=%MODEL_PATH:"=%/models/hiai
SET WinRarDir=%3
SET WinRarDir=%WinRarDir:"=%

cd /d %BASEPATH%

IF EXIST "%BASEPATH%/output" (
    rd /s /q output
)
md output

SET OUTPUT_PATH=%BASEPATH%/output

cd /d %TOOL_PATH%

IF EXIST tool (
    rd /s /q tool
)
md tool
"%WinRarDir%/WinRAR" x -l "%TOOL_PATH%/mindspore-lite-*-converter-win-cpu.zip" "%TOOL_PATH%/tool"
IF errorlevel 1 (
    echo "Decompression of converter tool is failed."
    call :run_fail
)

cd tool/mindspore-lite-*-converter-win-cpu/converter

converter_lite --outputFile="%OUTPUT_PATH%/detect" --modelFile="%MODEL_PATH%/detect.tflite" --fmk=TFLITE
IF errorlevel 1 (
    echo "Model conversion of detect.tflite is failed."
    call :run_fail
)

converter_lite --outputFile="%OUTPUT_PATH%/mobilenet_v1_0.5_128" --modelFile="%MODEL_PATH%/mobilenet_v1_0.5_128.tflite" --fmk=TFLITE
IF errorlevel 1 (
    echo "Model conversion of mobilenet_v1_0.5_128.tflite is failed."
    call :run_fail
)

converter_lite --outputFile="%OUTPUT_PATH%/mobilenet_v1_0.5_128_quant" --modelFile="%MODEL_PATH%/mobilenet_v1_0.5_128_quant.tflite" --fmk=TFLITE --quantType=AwareTraining
IF errorlevel 1 (
    echo "Model conversion of mobilenet_v1_0.5_128_quant.tflite is failed."
    call :run_fail
)

converter_lite --outputFile="%OUTPUT_PATH%/mtk_AADB_HADB_MBV2_model_f16" --modelFile="%MODEL_PATH%/mtk_AADB_HADB_MBV2_model_f16.tflite" --fmk=TFLITE
IF errorlevel 1 (
    echo "Model conversion of mtk_AADB_HADB_MBV2_model_f16.tflite is failed."
    call :run_fail
)

converter_lite --outputFile="%OUTPUT_PATH%/detect_mbv1_640_480_nopostprocess_simplified" --modelFile="%MODEL_PATH%/detect_mbv1_640_480_nopostprocess_simplified.onnx" --fmk=ONNX
IF errorlevel 1 (
    echo "Model conversion of detect_mbv1_640_480_nopostprocess_simplified.onnx is failed."
    call :run_fail
)

converter_lite --outputFile="%OUTPUT_PATH%/emotion" --modelFile="%MODEL_PATH%/emotion.prototxt" --weightFile="%MODEL_PATH%/emotion.caffemodel" --fmk=CAFFE
IF errorlevel 1 (
    echo "Model conversion of emotion.prototxt is failed."
    goto run_fail
)

SET /a count=0
FOR /f "tokens=* delims= " %%i in ('dir/s/b/a-d "%OUTPUT_PATH%\*.*"') DO (SET /a count=count+1)
IF NOT %count% == 6 (
    echo "Conversion of some models are failed."
    call :run_fail
)

:run_fail
    set errorlevel=1

:run_eof
    cd /d %BASEPATH%