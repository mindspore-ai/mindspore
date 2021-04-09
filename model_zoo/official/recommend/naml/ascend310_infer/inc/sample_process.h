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

#ifndef MINDSPORE_MODEL_ZOO_NAML_SAMPLE_PROCESS_H_
#define MINDSPORE_MODEL_ZOO_NAML_SAMPLE_PROCESS_H_

#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "./utils.h"
#include "./model_process.h"

class SampleProcess {
 public:
    SampleProcess();
    SampleProcess(uint32_t deviceId, uint32_t threadNum);

    ~SampleProcess();

    Result InitResource();

    Result Process(const std::vector<std::string> &omPaths,
                   const std::vector<std::string> &inputDataPaths,
                   const std::vector<std::string> &inputIdPaths,
                   const std::string &browsedNewsPath,
                   uint32_t batchSize);
    Result CreateModelProcessInstance(std::vector<std::string> omPaths, std::vector<std::string> inputDataPaths,
                                      std::vector<std::string> inputIdPaths, uint32_t batchSize);
    Result GetPred(uint32_t fileNum);
    int WriteResult(const std::string& imageFile, std::vector<float> result, uint32_t size);
    std::vector<std::string> GetModelExecCostTimeInfo();
    std::vector<std::vector<std::vector<int>>> ReadHistory(std::vector<std::string> historyFile, uint32_t batchSize);
    Result ReadBrowsedFile(const std::string &browsedNewsPath, std::vector<std::string> userIdFiles,
                           std::vector<std::vector<int>> *usersIds, std::vector<std::vector<int>> *candidateNewsIds);
    uint32_t ReadBrowsedData(const std::string &browsedNewsPath);
    void GetResult(uint32_t startPos, uint32_t endPos,
                                  std::map<int, void *> newsEncodeResult,
                                  std::map<int, void *> userEncodeResult);

 private:
    void DestroyResource();

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    std::map<int, std::shared_ptr<ModelProcess>> modelProcessContainer_;
    std::map<double, double> secondModelCostTime_map_;
    std::map<double, double> thirdModelCostTime_map_;
    std::map<double, double> totalCostTime_map_;
    std::vector<std::vector<int>> usersIds_;
    std::vector<std::vector<int>> candidateNewsIds_;
    std::vector<std::string> userIdFiles_;
    uint32_t threadNum_;
    std::mutex mtx_;
};

#endif
