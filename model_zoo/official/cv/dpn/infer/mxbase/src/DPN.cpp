/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
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

#include "DPN.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include <opencv2/dnn.hpp>
#include "MxBase/Log/Log.h"

using  namespace MxBase;
using  namespace cv::dnn;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t BATCH_SIZE = 32;
}  // namespace

APP_ERROR DPN::Init(const InitParam &initParam) {
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
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DPN::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DPN::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR DPN::ResizeImage(const cv::Mat &srcImageMat, cv::Mat *dstImageMat) {
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;

    cv::resize(srcImageMat, *dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR DPN::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }

    // mat NCHW to NHWC
    size_t N = 32, H = 224, W = 224, C = 3;
    unsigned char *mat_data = new unsigned char[dataSize];
    uint32_t idx = 0;
    for (size_t n = 0; n < N; n++) {
        int id = 0;
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                for (size_t c = 0; c < C; c++) {
                    id = n*(H*W*C) + c*(H*W) + h*W + w;
                    unsigned char *tmp = (unsigned char *)(imageMat.data + id);
                    mat_data[idx++] = *tmp;
                }
            }
        }
    }
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(mat_data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {32, 224, 224, 3};
    *tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR DPN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DPN::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<std::vector<MxBase::ClassInfo>> *clsInfos) {
    APP_ERROR ret = post_->Process(inputs, *clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DPN::SaveResult(const std::vector<std::string> &batchImgPaths,
                            const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos) {
    uint32_t batchIndex = 0;
    for (auto &imgPath : batchImgPaths) {
        std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
        size_t dot = fileName.find_last_of(".");
        std::string resFileName = "../results/" + fileName.substr(0, dot) + "_1.txt";
        std::ofstream outfile(resFileName);
        if (outfile.fail()) {
            LogError << "Failed to open result file: ";
            return APP_ERR_COMM_FAILURE;
        }
        auto clsInfos = batchClsInfos[batchIndex];
            std::string resultStr;
            for (auto clsInfo : clsInfos) {
                LogDebug << " className:" << clsInfo.className << " confidence:" << clsInfo.confidence <<
                " classIndex:" <<  clsInfo.classId;
                resultStr += std::to_string(clsInfo.classId) + " ";
            }
            outfile << resultStr << std::endl;
            batchIndex++;
        outfile.close();
    }
    return APP_ERR_OK;
}

APP_ERROR DPN::Process(const std::vector<std::string> &batchImgPaths) {
    APP_ERROR ret = APP_ERR_OK;
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    std::vector<cv::Mat> batchImgMat;
    for (auto &imgPath : batchImgPaths) {
        cv::Mat imageMat;
        ret = ReadImage(imgPath, &imageMat);
        if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        }
        ResizeImage(imageMat, &imageMat);
        batchImgMat.push_back(imageMat);
    }
    cv::Mat inputBlob = cv::dnn::blobFromImages(batchImgMat, 1.0f, cv::Size(), cv::Scalar(), false, false);
    inputBlob.convertTo(inputBlob, CV_8U);
    TensorBase tensorBase;
    ret = CVMatToTensorBase(inputBlob, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, &BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(batchImgPaths, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}



