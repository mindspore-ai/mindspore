/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef MODELPROCSS_H
#define MODELPROCSS_H

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <map>
#include <memory>
#include <string>
#include "acl/acl.h"
#include "CommonDataType.h"

class ModelProcess {
 public:
    explicit ModelProcess(const int deviceId);
    ModelProcess();
    ~ModelProcess();

    int Init(const std::string &modelPath);
    int DeInit();

    int ModelInference(const std::vector<void *> &inputBufs,
                       const std::vector<size_t> &inputSizes,
                       const std::vector<void *> &ouputBufs,
                       const std::vector<size_t> &outputSizes,
                       std::map<double, double> *costTime_map);
    aclmdlDesc *GetModelDesc();
    int ReadBinaryFile(const std::string &fileName, uint8_t **buffShared, int *buffLength);

 private:
    aclmdlDataset *CreateAndFillDataset(const std::vector<void *> &bufs, const std::vector<size_t> &sizes);
    void DestroyDataset(aclmdlDataset *dataset);

    std::mutex mtx_ = {};
    int deviceId_ = 0;
    uint32_t modelId_ = 0;
    void *modelDevPtr_ = nullptr;
    size_t modelDevPtrSize_ = 0;
    void *weightDevPtr_ = nullptr;
    size_t weightDevPtrSize_ = 0;
    aclrtContext contextModel_ = nullptr;
    std::shared_ptr<aclmdlDesc> modelDesc_ = nullptr;
    bool isDeInit_ = false;
};

#endif
