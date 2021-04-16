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
@title win_run_test
setlocal enabledelayedexpansion

SET PACKAGE_PATH=%1
SET MODEL_PATH_BASE=%2
SET INSTRUCTION=%3

SET BASEPATH=%CD%
SET RET_CODE=0

SET PACKAGE_PATH=%PACKAGE_PATH:"=%\windows_x64\%INSTRUCTION%
7z x -r "%PACKAGE_PATH%\mindspore-lite-*.zip"
IF NOT %errorlevel% == 0 (
    echo "Decompression of runtime tool fail!"
    SET RET_CODE=1
    goto run_eof
)

for /f %%i in ('dir /b %PACKAGE_PATH%\mindspore-lite-*.zip') do set PACKAGE_NAME=%%i
set PACKAGE_NAME=%PACKAGE_NAME:.zip=%
SET DST_PACKAGE_PATH=%BASEPATH%\%PACKAGE_NAME%

echo "Convert models"
copy %DST_PACKAGE_PATH%\tools\converter\lib\* %DST_PACKAGE_PATH%\tools\converter\converter\
cd /d %DST_PACKAGE_PATH%\tools\converter\converter\

SET TYPE_ID=''
SET MODEL_NAME=''
SET SUFFIX=''
SET MODEL_CONFIG=%BASEPATH%\win_models.cfg
SET MODEL_PATH=%MODEL_PATH_BASE:"=%\models\hiai

for /f "tokens=1-2 delims= " %%i in (%MODEL_CONFIG%) do (
    for /f "tokens=1-2 delims=." %%k in ("%%j") do (
        SET TYPE_ID=%%i
        SET MODEL_NAME=%%k
        SET SUFFIX=%%l
    )
    if !TYPE_ID!==1 (converter_lite --fmk=MINDIR --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!")
    if !TYPE_ID!==2 (converter_lite --fmk=MINDIR --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_train" --trainModel=true)
    if !TYPE_ID!==3 (converter_lite --fmk=CAFFE --modelFile="%MODEL_PATH%/!MODEL_NAME!.prototxt" --weightFile="%MODEL_PATH%/!MODEL_NAME!.caffemodel" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!")
    if !TYPE_ID!==4 (converter_lite --fmk=ONNX --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!")
    if !TYPE_ID!==5 (converter_lite --fmk=TFLITE --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!")
    if !TYPE_ID!==6 (converter_lite --fmk=TFLITE --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!" --quantType=AwareTraining)
    if !TYPE_ID!==7 (converter_lite --fmk=TFLITE --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_posttraining" --quantType=PostTraining --configFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!_posttraining.config")
    if !TYPE_ID!==8 (converter_lite --fmk=TFLITE --modelFile="%MODEL_PATH%/!MODEL_NAME!.!SUFFIX!" --outputFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_weightquant" --quantType=WeightQuant --bitNum=8 --quantWeightSize=500 --quantWeightChannel=16)

    IF !errorlevel! == 0 (
        echo "Run converter for model (TYPE_ID=!TYPE_ID!) !MODEL_NAME!.!SUFFIX! : pass!"
    ) ELSE (
        echo "Run converter for model (TYPE_ID=!TYPE_ID!) !MODEL_NAME!.!SUFFIX! : fail!"
        SET RET_CODE=1
        goto run_eof
    )
)

echo "Run converted models"
copy %DST_PACKAGE_PATH%\inference\lib\* %DST_PACKAGE_PATH%\tools\benchmark\
cd /d %DST_PACKAGE_PATH%\tools\benchmark\

SET INPUT_BASE=%MODEL_PATH%/input_output/input
SET OUTPUT_BASE=%MODEL_PATH%/input_output/output

for /f "tokens=1-2 delims= " %%i in (%MODEL_CONFIG%) do (
    for /f "tokens=1-2 delims=." %%k in ("%%j") do (
        SET TYPE_ID=%%i
        SET MODEL_NAME=%%k
        SET SUFFIX=%%l
    )
    if !TYPE_ID!==1 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out" --accuracyThreshold=1.5)
    if !TYPE_ID!==2 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_train.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out" --accuracyThreshold=1.5)
    if !TYPE_ID!==3 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.ms.out")
    if !TYPE_ID!==4 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out")
    if !TYPE_ID!==5 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out")
    if !TYPE_ID!==6 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out")
    if !TYPE_ID!==7 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_posttraining.ms" --inDataFile="%MODEL_PATH_BASE%/quantTraining/mnist_calibration_data/00099.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!_posttraining.ms.out")
    if !TYPE_ID!==8 (benchmark --modelFile="%DST_PACKAGE_PATH%\!MODEL_NAME!.!SUFFIX!_weightquant.ms" --inDataFile="%INPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.bin" --benchmarkDataFile="%OUTPUT_BASE%/!MODEL_NAME!.!SUFFIX!.ms.out")

    IF !errorlevel! == 0 (
        echo "Run benchmark for model (TYPE_ID=!TYPE_ID!) !MODEL_NAME!.!SUFFIX! : pass!"
    ) ELSE (
        echo "Run benchmark for model (TYPE_ID=!TYPE_ID!) !MODEL_NAME!.!SUFFIX! : fail!"
        SET RET_CODE=1
        goto run_eof
    )
)

:run_eof
    cd /d %BASEPATH%
    rd /s /q %PACKAGE_NAME%
    IF %RET_CODE% == 0 (
        SET errorlevel=0
        echo "Run models in Windows success!"
    ) ELSE (
        SET errorlevel=1
        echo "Run models in Windows fail!"
    )
    exit /b %errorlevel%
