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

#ifndef MINDSPORE_MODEL_ZOO_NAML_MODEL_PROCESS_H_
#define MINDSPORE_MODEL_ZOO_NAML_MODEL_PROCESS_H_
#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "./utils.h"
#include "acl/acl.h"

class ModelProcess {
 public:
    ModelProcess() = default;
    ModelProcess(const std::string &inputDataPath, const std::string &idFilePath, uint32_t batchSize);

    ~ModelProcess();

    Result LoadModelFromFileWithMem(const char *modelPath);

    void Unload();

    Result CreateDesc();

    void DestroyDesc();

    void DestroyInput();

    Result CreateOutput();

    void DestroyOutput();

    Result Execute(uint32_t index);

    void OutputModelResult();
    Result CreateInput();
    Result CpyFileToDevice(std::string fileName, uint32_t inputNum);
    void CpyOutputFromDeviceToHost(uint32_t index);
    std::map<int, void *> GetResult();
    std::vector<uint32_t> GetOutputSize();
    std::vector<uint32_t> GetInputSize();
    Result ExecuteWithFile(uint32_t fileNum);
    Result CpyDataToDevice(void *data, uint32_t len, uint32_t inputNum);
    std::string GetInputDataPath();
    std::string GetCostTimeInfo();
    void DestroyResource();
    std::vector<std::vector<void *>> ReadInputFiles(std::vector<std::vector<std::string>> inputFiles,
                                                    size_t inputSize, std::vector<std::vector<uint32_t>> *fileSize);
    Result ReadIdFiles();
    Result InitResource();
    uint32_t ReadFiles();

 private:
    uint32_t modelId_;
    std::map<double, double> costTime_map_;
    size_t modelMemSize_;
    size_t modelWeightSize_;
    void *modelMemPtr_;
    void *modelWeightPtr_;
    uint32_t batchSize_;
    bool loadFlag_;  // model load flag
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    uint32_t inputNum_;
    std::vector<uint32_t> inputBuffSize_;
    aclmdlDataset *output_;
    uint32_t outputNum_;
    std::vector<uint32_t> outputBuffSize_;

    std::map<int, void *> result_;
    std::vector<void *> resultMem_;
    std::vector<void *> fileBuffMem_;
    std::string inputDataPath_;
    std::string idFilePath_;
    std::vector<std::vector<void *>> fileBuff_;
    std::vector<std::vector<uint32_t>> fileSize_;
    std::vector<std::vector<int>> ids_;
};

#endif
