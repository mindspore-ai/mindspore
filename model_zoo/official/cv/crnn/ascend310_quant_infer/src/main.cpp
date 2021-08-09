/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "../inc/sample_process.h"
#include "../inc/utils.h"
bool g_is_device = false;

int main(int argc, char **argv) {
    if (argc != 3) {
        ERROR_LOG("usage:./main path_of_om path_of_inputFolder");
        return FAILED;
    }
    SampleProcess processSample;
    Result ret = processSample.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }

    ret = processSample.Process(argv[1], argv[2]);
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }

    INFO_LOG("execute sample success");
    return SUCCESS;
}
