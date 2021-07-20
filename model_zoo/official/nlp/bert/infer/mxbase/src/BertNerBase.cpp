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
#include "BertNerBase.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t EACH_LABEL_LENGTH = 4;
const uint32_t MAX_LENGTH = 128;
const uint32_t CLASS_NUM = 41;

APP_ERROR BertNerBase::LoadLabels(const std::string &labelPath, std::vector<std::string> *labelMap) {
    std::ifstream infile;
    // open label file
    infile.open(labelPath, std::ios_base::in);
    std::string s;
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << labelPath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    labelMap->clear();
    // construct label vector
    while (std::getline(infile, s)) {
        if (s.size() == 0 || s[0] == '#') {
            continue;
        }
        size_t eraseIndex = s.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos) {
            s.erase(eraseIndex + 1, s.size() - eraseIndex);
        }
        labelMap->push_back(s);
    }
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    classNum_ = initParam.classNum;
    // load labels from file
    ret = LoadLabels(initParam.labelPath, &labelMap_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size) {
    if (data == NULL || size < MAX_LENGTH) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::ifstream infile;
    // open label file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(uint32_t) * MAX_LENGTH);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::ReadInputTensor(const std::string &fileName, uint32_t index,
                                       std::vector<MxBase::TensorBase> *inputs) {
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, MAX_LENGTH};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<uint32_t> *argmax) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        std::vector<float> result = {};
        for (uint32_t j = 0; j < classNum; j++) {
            float value = *(reinterpret_cast<float*>(data) + i * classNum + j);
            result.push_back(value);
        }
        // argmax and get the class id
        std::vector<float>::iterator maxElement = std::max_element(std::begin(result), std::end(result));
        uint32_t argmaxIndex = maxElement - std::begin(result);
        argmax->push_back(argmaxIndex);
    }

    return APP_ERR_OK;
}

APP_ERROR BertNerBase::CountPredictResult(const std::string &labelFile, const std::vector<uint32_t> &argmax) {
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = ReadTensorFromFile(labelFile, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    uint32_t target[CLASS_NUM][MAX_LENGTH] = {0};
    uint32_t pred[CLASS_NUM][MAX_LENGTH] = {0};
    for (uint32_t i = 0; i < MAX_LENGTH; i++) {
        if (data[i] > 0) {
            target[data[i]][i] = 1;
        }
        if (argmax[i] > 0) {
            pred[argmax[i]][i] = 1;
        }
    }
    for (uint32_t i = 0; i < CLASS_NUM; i++) {
        for (uint32_t j = 0; j < MAX_LENGTH; j++) {
            // count True Positive and False Positive
            if (pred[i][j] == 1) {
                if (target[i][j] == 1) {
                    g_TP += 1;
                } else {
                    g_FP += 1;
                }
            }
            // count False Negative
            if (target[i][j] == 1 && pred[i][j] != 1) {
                g_FN += 1;
            }
        }
    }
    LogInfo << "TP: " << g_TP << ", FP: " << g_FP << ", FN: " << g_FN;
    return APP_ERR_OK;
}

void BertNerBase::GetClunerLabel(const std::vector<uint32_t> &argmax, std::multimap<std::string,
                                 std::vector<uint32_t>> *clunerMap) {
    bool findCluner = false;
    uint32_t start = 0;
    std::string clunerName;
    for (uint32_t i = 0; i < argmax.size(); i++) {
        if (argmax[i] > 0) {
            if (!findCluner) {
                start = i;
                clunerName = labelMap_[(argmax[i] - 1) / EACH_LABEL_LENGTH];
                findCluner = true;
            } else {
                if (labelMap_[(argmax[i] - 1) / EACH_LABEL_LENGTH] != clunerName) {
                    std::vector<uint32_t> position = {start - 1, i - 2};
                    clunerMap->insert(std::pair<std::string, std::vector<uint32_t>>(clunerName, position));
                    start = i;
                    clunerName = labelMap_[(argmax[i] - 1) / EACH_LABEL_LENGTH];
                }
            }
        } else {
            if (findCluner) {
                std::vector<uint32_t> position = {start - 1, i - 2};
                clunerMap->insert(std::pair<std::string, std::vector<uint32_t>>(clunerName, position));
                findCluner = false;
            }
        }
    }
}

APP_ERROR BertNerBase::WriteResult(const std::string &fileName, const std::vector<uint32_t> &argmax) {
    std::string resultPathName = "result";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/result.txt";
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    LogInfo << "==============================================================";
    LogInfo << "infer result of " << fileName << " is: ";
    tfile << "file name is: " << fileName << std::endl;
    std::multimap<std::string, std::vector<uint32_t>> clunerMap;
    GetClunerLabel(argmax, &clunerMap);
    for (auto &item : clunerMap) {
        LogInfo << item.first << ": " << item.second[0] << ", " << item.second[1];
        tfile << item.first << ": " << item.second[0] << ", " << item.second[1] << std::endl;
    }
    LogInfo << "==============================================================";
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR BertNerBase::Process(const std::string &inferPath, const std::string &fileName, bool eval) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = inferPath + "00_data/" + fileName;
    APP_ERROR ret = ReadInputTensor(inputIdsFile, INPUT_IDS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }
    std::string inputMaskFile = inferPath + "01_data/" + fileName;
    ret = ReadInputTensor(inputMaskFile, INPUT_MASK, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input mask file failed, ret=" << ret << ".";
        return ret;
    }
    std::string tokenTypeIdFile = inferPath + "02_data/" + fileName;
    ret = ReadInputTensor(tokenTypeIdFile, TOKEN_TYPE, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read token typeId file failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<uint32_t> argmax;
    ret = PostProcess(&outputs, &argmax);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(fileName, argmax);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }

    if (eval) {
        std::string labelFile = inferPath + "03_data/" + fileName;
        ret = CountPredictResult(labelFile, argmax);
        if (ret != APP_ERR_OK) {
            LogError << "CalcF1Score read label failed, ret=" << ret << ".";
            return ret;
        }
    }

    return APP_ERR_OK;
}
