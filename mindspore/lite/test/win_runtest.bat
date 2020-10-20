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

SET BASEPATH=%CD%

call run_benchmark_nets.bat %1 %2
IF NOT %errorlevel% == 0 (
    echo "benchmark fail!"
    SET errorlevel=1
) ELSE (
    echo "run benchmark tests success."
)

cd /d %BASEPATH%
