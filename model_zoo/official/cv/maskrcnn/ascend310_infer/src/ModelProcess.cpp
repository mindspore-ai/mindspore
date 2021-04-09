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

#include <sys/time.h>
#include <fstream>
#include "../inc/ModelProcess.h"

ModelProcess::ModelProcess(const int deviceId) {
    deviceId_ = deviceId;
}

ModelProcess::ModelProcess() {}

ModelProcess::~ModelProcess() {
    if (!isDeInit_) {
        DeInit();
    }
}

void ModelProcess::DestroyDataset(aclmdlDataset *dataset) {
    // Just release the DataBuffer object and DataSet object, remain the buffer, because it is managerd by user
    if (dataset != nullptr) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); i++) {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
            if (dataBuffer != nullptr) {
                aclDestroyDataBuffer(dataBuffer);
                dataBuffer = nullptr;
            }
        }
        aclmdlDestroyDataset(dataset);
    }
}

aclmdlDesc *ModelProcess::GetModelDesc() {
    return modelDesc_.get();
}

int ModelProcess::ModelInference(const std::vector<void *> &inputBufs,
                                 const std::vector<size_t> &inputSizes,
                                 const std::vector<void *> &ouputBufs,
                                 const std::vector<size_t> &outputSizes,
                                 std::map<double, double> *costTime_map) {
    std::cout << "ModelProcess:Begin to inference." << std::endl;
    aclmdlDataset *input = nullptr;
    input = CreateAndFillDataset(inputBufs, inputSizes);
    if (input == nullptr) {
        return INVALID_POINTER;
    }
    int ret;

    aclmdlDataset *output = nullptr;
    output = CreateAndFillDataset(ouputBufs, outputSizes);
    if (output == nullptr) {
        DestroyDataset(input);
        input = nullptr;
        return INVALID_POINTER;
    }
    struct timeval start;
    struct timeval end;
    double startTime_ms;
    double endTime_ms;
    mtx_.lock();
    gettimeofday(&start, NULL);
    ret = aclmdlExecute(modelId_, input, output);
    gettimeofday(&end, NULL);
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map->insert(std::pair<double, double>(startTime_ms, endTime_ms));
    mtx_.unlock();
    if (ret != OK) {
        std::cout << "aclmdlExecute failed, ret[" << ret << "]." << std::endl;
        return ret;
    }

    DestroyDataset(input);
    DestroyDataset(output);
    return OK;
}

int ModelProcess::DeInit() {
    isDeInit_ = true;
    int ret = aclmdlUnload(modelId_);
    if (ret != OK) {
        std::cout << "aclmdlUnload  failed, ret["<< ret << "]." << std::endl;
        return ret;
    }

    if (modelDevPtr_ != nullptr) {
        ret = aclrtFree(modelDevPtr_);
        if (ret != OK) {
            std::cout << "aclrtFree  failed, ret[" << ret << "]." << std::endl;
            return ret;
        }
        modelDevPtr_ = nullptr;
    }
    if (weightDevPtr_ != nullptr) {
        ret = aclrtFree(weightDevPtr_);
        if (ret != OK) {
            std::cout << "aclrtFree  failed, ret[" << ret << "]." << std::endl;
            return ret;
        }
        weightDevPtr_ = nullptr;
    }

    return OK;
}
/**
 * Read a binary file, store the data into a uint8_t array
 *
 * @param fileName the file for reading
 * @param buffShared a shared pointer to a uint8_t array for storing file
 * @param buffLength the length of the array
 * @return OK if create success, error code otherwise
 */
int ModelProcess::ReadBinaryFile(const std::string &fileName, uint8_t **buffShared, int *buffLength) {
    std::ifstream inFile(fileName, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cout << "FaceFeatureLib: read file " << fileName << " fail." <<std::endl;
        return READ_FILE_FAIL;
    }

    inFile.seekg(0, inFile.end);
    *buffLength = inFile.tellg();
    inFile.seekg(0, inFile.beg);

    uint8_t *tempShared = reinterpret_cast<uint8_t *>(malloc(*buffLength));
    inFile.read(reinterpret_cast<char *>(tempShared), *buffLength);
    inFile.close();
    *buffShared = tempShared;

    std::cout << "read file: fileName=" << fileName << ", size=" << *buffLength << "." << std::endl;

    return OK;
}

int ModelProcess::Init(const std::string &modelPath) {
    std::cout << "ModelProcess:Begin to init instance." << std::endl;
    int modelSize = 0;
    uint8_t *modelData = nullptr;
    int ret = ReadBinaryFile(modelPath, &modelData, &modelSize);
    if (ret != OK) {
        std::cout << "read model file failed, ret[" << ret << "]." << std::endl;
        return ret;
    }
    ret = aclmdlQuerySizeFromMem(modelData, modelSize, &modelDevPtrSize_, &weightDevPtrSize_);
    if (ret != OK) {
        std::cout << "aclmdlQuerySizeFromMem failed, ret[" << ret << "]." << std::endl;
        return ret;
    }
    std::cout << "modelDevPtrSize_[" << modelDevPtrSize_ << "]" << std::endl;
    std::cout << " weightDevPtrSize_[" << weightDevPtrSize_ << "]." << std::endl;

    ret = aclrtMalloc(&modelDevPtr_, modelDevPtrSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != OK) {
        std::cout << "aclrtMalloc dev_ptr failed, ret[" << ret << "]." << std::endl;
        return ret;
    }
    ret = aclrtMalloc(&weightDevPtr_, weightDevPtrSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != OK) {
        std::cout << "aclrtMalloc weight_ptr failed, ret[" << ret << "] " << std::endl;
        return ret;
    }
    ret = aclmdlLoadFromMemWithMem(modelData, modelSize, &modelId_, modelDevPtr_, modelDevPtrSize_,
        weightDevPtr_, weightDevPtrSize_);
    if (ret != OK) {
        std::cout << "aclmdlLoadFromMemWithMem failed, ret[" << ret << "]." << std::endl;
        return ret;
    }
    ret = aclrtGetCurrentContext(&contextModel_);
    if (ret != OK) {
        std::cout << "aclrtMalloc weight_ptr failed, ret[" << ret << "]." << std::endl;
        return ret;
    }

    aclmdlDesc *modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
        std::cout << "aclmdlCreateDesc failed." << std::endl;
        return ret;
    }
    ret = aclmdlGetDesc(modelDesc, modelId_);
    if (ret != OK) {
        std::cout << "aclmdlGetDesc ret fail, ret:" << ret << "." << std::endl;
        return ret;
    }
    modelDesc_.reset(modelDesc, aclmdlDestroyDesc);
    free(modelData);
    return OK;
}

aclmdlDataset *ModelProcess::CreateAndFillDataset(const std::vector<void *> &bufs, const std::vector<size_t> &sizes) {
    aclmdlDataset *dataset = aclmdlCreateDataset();
    if (dataset == nullptr) {
        std::cout << "ACL_ModelInputCreate failed." << std::endl;
        return nullptr;
    }

    for (size_t i = 0; i < bufs.size(); ++i) {
        aclDataBuffer *data = aclCreateDataBuffer(bufs[i], sizes[i]);
        if (data == nullptr) {
            DestroyDataset(dataset);
            std::cout << "aclCreateDataBuffer failed." << std::endl;
            return nullptr;
        }

        int ret = aclmdlAddDatasetBuffer(dataset, data);
        if (ret != OK) {
            DestroyDataset(dataset);
            std::cout << "ACL_ModelInputDataAdd failed, ret[" << ret << "]." << std::endl;
            return nullptr;
        }
    }
    return dataset;
}
