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

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <algorithm>
#include "../inc/utils.h"
#include "../inc/model_process.h"

extern bool g_isDevice;

ModelProcess::ModelProcess(const std::string &inputDataPath, const std::string &idFilePath, uint32_t batchSize):
    modelId_(0),
    modelMemSize_(0),
    modelWeightSize_(0),
    modelMemPtr_(nullptr),
    modelWeightPtr_(nullptr),
    loadFlag_(false),
    modelDesc_(nullptr),
    output_(nullptr),
    inputDataPath_(inputDataPath),
    input_(nullptr),
    batchSize_(batchSize),
    idFilePath_(idFilePath),
    inputNum_(0),
    outputNum_(0) {}

ModelProcess::~ModelProcess() {
    Unload();
    DestroyResource();
}

Result ModelProcess::InitResource() {
    Result ret = CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    ret = CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("create model output failed");
        return FAILED;
    }

    ret = CreateInput();
    if (ret != SUCCESS) {
        ERROR_LOG("create model input failed");
        return FAILED;
    }

    ret = ReadIdFiles();
    if (ret != SUCCESS) {
        ERROR_LOG("read id files failed");
        return FAILED;
    }
}
void ModelProcess::DestroyResource() {
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
    for (auto addr : resultMem_) {
        if (addr != nullptr) {
            aclrtFreeHost(addr);
        }
    }

    for (auto addr : fileBuffMem_) {
        if (addr != nullptr) {
            aclrtFreeHost(addr);
        }
    }
    return;
}

Result ModelProcess::LoadModelFromFileWithMem(const char *modelPath) {
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return FAILED;
    }

    aclError ret = aclmdlQuerySize(modelPath, &modelMemSize_, &modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("query model failed, model file is %s", modelPath);
        return FAILED;
    }

    ret = aclrtMalloc(&modelMemPtr_, modelMemSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for mem failed, require size is %zu", modelMemSize_);
        return FAILED;
    }

    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu", modelWeightSize_);
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelMemPtr_,
        modelMemSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("load model from file failed, model file is %s", modelPath);
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::CreateDesc() {
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc() {
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::CreateInput() {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create output failed");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }

    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    inputNum_ = inputSize;
    for (size_t i = 0; i < inputSize; ++i) {
        size_t buffer_size = aclmdlGetInputSizeByIndex(modelDesc_, i);
        inputBuffSize_.emplace_back(buffer_size);

        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create input failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, buffer_size);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't create data buffer, create input failed");
            aclrtFree(inputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(inputBuffer);
            aclDestroyDataBuffer(inputData);
            return FAILED;
        }
    }

    INFO_LOG("create model input success");
    return SUCCESS;
}

void ModelProcess::DestroyInput() {
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(input_);
    input_ = nullptr;
}

Result ModelProcess::CreateOutput() {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create output failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    outputNum_ = outputSize;
    for (size_t i = 0; i < outputSize; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        outputBuffSize_.emplace_back(buffer_size);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

void ModelProcess::OutputModelResult() {
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = NULL;
        aclError ret = ACL_ERROR_NONE;
        float *outData = NULL;
        if (!g_isDevice) {
            ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMallocHost failed, ret[%d]", ret);
                return;
            }

            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
                return;
            }

            outData = reinterpret_cast<float *>(outHostData);
        } else {
            outData = reinterpret_cast<float *>(data);
        }
        std::map<float, unsigned int, std::greater<float> > resultMap;
        for (unsigned int j = 0; j < len / sizeof(float); ++j) {
            resultMap[*outData] = j;
            outData++;
        }

        int cnt = 0;
        for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
            // print top 5
            if (++cnt > 5) {
                break;
            }

            INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);
        }
        if (!g_isDevice) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtFreeHost failed, ret[%d]", ret);
                return;
            }
        }
    }

    INFO_LOG("output data success");
    return;
}

void ModelProcess::DestroyOutput() {
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

Result ModelProcess::CpyFileToDevice(std::string fileName, uint32_t inputNum) {
    uint32_t inputHostBuffSize = 0;
    void *inputHostBuff = Utils::ReadBinFile(fileName, &inputHostBuffSize);
    if (inputHostBuff == nullptr) {
        return FAILED;
    }
    aclDataBuffer *inBufferDev = aclmdlGetDatasetBuffer(input_, inputNum);
    void *p_batchDst = aclGetDataBufferAddr(inBufferDev);
    aclrtMemset(p_batchDst, inputHostBuffSize, 0, inputHostBuffSize);
    aclError ret = aclrtMemcpy(p_batchDst, inputHostBuffSize, inputHostBuff, inputHostBuffSize,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                  inputHostBuffSize, inputHostBuffSize);
        aclrtFreeHost(inputHostBuff);
        return FAILED;
    }
    aclrtFreeHost(inputHostBuff);
    return SUCCESS;
}

Result ModelProcess::CpyDataToDevice(void *data, uint32_t len, uint32_t inputNum) {
    if (len != inputBuffSize_[inputNum]) {
        return FAILED;
    }
    aclDataBuffer *inBufferDev = aclmdlGetDatasetBuffer(input_, inputNum);
    void *p_batchDst = aclGetDataBufferAddr(inBufferDev);
    aclrtMemset(p_batchDst, len, 0, len);
    aclError ret = aclrtMemcpy(p_batchDst, len, data, len, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                  len, len);
        return FAILED;
    }
    return SUCCESS;
}

void ModelProcess::CpyOutputFromDeviceToHost(uint32_t index) {
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);

    for (size_t i = 0; i < outputNum; ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t bufferSize = aclGetDataBufferSizeV2(dataBuffer);

        void* outHostData = NULL;
        aclError ret = ACL_ERROR_NONE;
        ret = aclrtMallocHost(&outHostData, bufferSize);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrtMallocHost failed, ret[%d]", ret);
            return;
        }
        resultMem_.emplace_back(outHostData);
        ret = aclrtMemcpy(outHostData, bufferSize, data, bufferSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
            (void)aclrtFreeHost(outHostData);
            return;
        }

        uint32_t len = (uint32_t)bufferSize / batchSize_;
        for (size_t j = 0; j < batchSize_; j++) {
            result_.emplace(ids_[index][j], reinterpret_cast<uint8_t *>(outHostData) + (j * len));
        }
    }
    return;
}

std::vector<std::vector<void *>> ModelProcess::ReadInputFiles(std::vector<std::vector<std::string>> inputFiles,
                                                              size_t inputSize,
                                                              std::vector<std::vector<uint32_t>> *fileSize) {
    size_t fileNum = inputFiles[0].size();
    std::vector<std::vector<void *>> buff(fileNum);
    if (inputFiles.size() != inputSize) {
        std::cout << "the num of input file is incorrect" << std::endl;
        return buff;
    }

    void *inputHostBuff = nullptr;
    uint32_t inputHostBuffSize = 0;
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < fileNum; ++j) {
            inputHostBuff = Utils::ReadBinFile(inputFiles[i][j], &inputHostBuffSize);
            buff[i].emplace_back(inputHostBuff);
            fileBuffMem_.emplace_back(inputHostBuff);
            (*fileSize)[i].emplace_back(inputHostBuffSize);
        }
    }

    return buff;
}

Result ModelProcess::ReadIdFiles() {
    std::vector<std::string> idFiles = Utils::GetAllBins(idFilePath_);

    for (int i = 0; i < idFiles.size(); ++i) {
        std::vector<int> ids;
        Utils::ReadFileToVector(idFiles[i], batchSize_, &ids);
        ids_.emplace_back(ids);
    }
    return SUCCESS;
}

uint32_t ModelProcess::ReadFiles() {
    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    std::vector<std::vector<uint32_t>> fileSize(inputSize);
    std::vector<std::vector<std::string>> inputFiles = Utils::GetAllInputData(inputDataPath_);

    fileBuff_ = ReadInputFiles(inputFiles, inputSize, &fileSize);
    uint32_t fileNum = inputFiles[0].size();
    fileSize_ = fileSize;
    return fileNum;
}

Result ModelProcess::ExecuteWithFile(uint32_t fileNum) {
    for (size_t index = 0; index < fileNum; ++index) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;
        gettimeofday(&start, NULL);
        void *picDevBuffer = nullptr;

        for (auto i = 0; i < inputNum_; ++i) {
            CpyDataToDevice(fileBuff_[i][index], fileSize_[i][index], i);
        }

        aclError ret = aclmdlExecute(modelId_, input_, output_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("execute model failed, modelId is %u", modelId_);
            return FAILED;
        }

        CpyOutputFromDeviceToHost(index);
        gettimeofday(&end, NULL);
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map_.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    }
    return SUCCESS;
}

Result ModelProcess::Execute(uint32_t index) {
    struct timeval start;
    struct timeval end;
    double startTime_ms;
    double endTime_ms;

    gettimeofday(&start, NULL);
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }

    CpyOutputFromDeviceToHost(index);
    gettimeofday(&end, NULL);
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map_.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    return SUCCESS;
}

std::map<int, void *> ModelProcess::GetResult() {
    return result_;
}

void ModelProcess::Unload() {
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}

std::vector<uint32_t> ModelProcess::GetInputSize() {
    return inputBuffSize_;
}

std::vector<uint32_t> ModelProcess::GetOutputSize() {
    return outputBuffSize_;
}

std::string ModelProcess::GetInputDataPath() {
    return inputDataPath_;
}

std::string ModelProcess::GetCostTimeInfo() {
    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map_.begin(); iter != costTime_map_.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }
    average = average / infer_cnt;

    std::stringstream timeCost;
    timeCost << "first model latency "<< average << " ms; count " << infer_cnt << std::endl;

    return timeCost.str();
}
