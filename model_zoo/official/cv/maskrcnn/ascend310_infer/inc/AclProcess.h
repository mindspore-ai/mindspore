/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ACLMANAGER_H
#define ACLMANAGER_H

#include <map>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "CommonDataType.h"
#include "ModelProcess.h"
#include "DvppCommon.h"

struct ModelInfo {
    std::string modelPath;
    uint32_t modelWidth;
    uint32_t modelHeight;
    uint32_t outputNum;
};

class AclProcess {
 public:
    AclProcess(int deviceId, const std::string &om_path, uint32_t width, uint32_t height);
    ~AclProcess() {}
    void Release();
    int InitResource();
    int Process(const std::string& imageFile, std::map<double, double> *costTime_map);

 private:
    int InitModule();
    int Preprocess(const std::string& imageFile);
    int ModelInfer(std::map<double, double> *costTime_map);
    int WriteResult(const std::string& imageFile);
    int ReadFile(const std::string &filePath, RawData *fileData);

    int32_t deviceId_;
    ModelInfo modelInfo_;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> modelProcess_;
    std::shared_ptr<DvppCommon> dvppCommon_;
    bool keepRatio_;
    std::vector<void *> outputBuffers_;
    std::vector<size_t> outputSizes_;
};

#endif
