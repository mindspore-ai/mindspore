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

#ifndef MINDSPORE_MODEL_ZOO_NAML_UTILS_H_
#define MINDSPORE_MODEL_ZOO_NAML_UTILS_H_

#pragma once
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class Utils {
 public:
    static void *GetDeviceBufferOfFile(std::string fileName, uint32_t *fileSize);

    static void *ReadBinFile(std::string fileName, uint32_t *fileSize);

    static std::vector <std::vector<std::string>> GetAllInputData(std::string dir_name);

    static DIR *OpenDir(std::string dir_name);

    static std::string RealPath(std::string path);

    static std::vector <std::string> GetAllBins(std::string dir_name);

    static Result ReadFileToVector(std::string newsIdFileName, uint32_t batchSize, std::vector<int> *newsId);
    static Result ReadFileToVector(std::string newsIdFileName, std::vector<int> *newsId);
    static Result ReadFileToVector(std::string newsIdFileName, uint32_t batchSize, uint32_t count,
                                   std::vector<std::vector<int>> *newsId);
};
#pragma once

#endif
