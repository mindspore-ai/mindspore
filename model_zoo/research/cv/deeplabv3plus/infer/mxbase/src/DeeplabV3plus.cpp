/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "DeeplabV3plus.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using std::max;
using std::vector;

const float IMAGE_SIZE = 513;

APP_ERROR DeeplabV3plus::Init(const InitParam &initParam) {
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
    MxBase::ConfigData configData;
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("CHECK_MODEL", std::to_string(initParam.checkModel));
    configData.SetJsonValue("FRAMEWORK_TYPE", std::to_string(initParam.frameworkType));

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
    post_ = std::make_shared<MxBase::Deeplabv3Post>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Deeplabv3plusPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

void DeeplabV3plus::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

void DeeplabV3plus::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
}

void DeeplabV3plus::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
    MxBase::ResizedImageInfo &resizedImageInfo) {
    float scale = IMAGE_SIZE / max(srcImageMat.rows, srcImageMat.cols);
    int dstWidth = srcImageMat.cols * scale;
    int dstHeight = srcImageMat.rows * scale;
    cv::resize(srcImageMat, dstImageMat, cv::Size(dstWidth, dstHeight));

    resizedImageInfo.heightOriginal = srcImageMat.rows;
    resizedImageInfo.heightResize = dstWidth;
    resizedImageInfo.widthOriginal = srcImageMat.cols;
    resizedImageInfo.widthResize = dstHeight;
    resizedImageInfo.resizeType = MxBase::RESIZER_MS_KEEP_ASPECT_RATIO;
}

void DeeplabV3plus::Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    constexpr size_t ALPHA_AND_BETA_SIZE = 3;
    cv::Mat float32Mat;
    srcImageMat.convertTo(float32Mat, CV_32FC3);

    std::vector<cv::Mat> tmp;
    cv::split(float32Mat, tmp);

    const std::vector<double> mean = {103.53, 116.28, 123.675};
    const std::vector<double> std = {57.375, 57.120, 58.395};
    for (size_t i = 0; i < ALPHA_AND_BETA_SIZE; ++i) {
        tmp[i].convertTo(tmp[i], CV_32FC3, 1 / std[i], - mean[i] / std[i]);
    }
    cv::merge(tmp, dstImageMat);
}

void DeeplabV3plus::Padding(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    constexpr int32_t paddingHeight = 513;
    constexpr int32_t paddingWidth = 513;
    int padH = paddingHeight - srcImageMat.rows;
    int padW = paddingWidth - srcImageMat.cols;
    if (padH > 0 || padW > 0) {
        cv::Scalar value(0, 0, 0);
        cv::copyMakeBorder(srcImageMat, dstImageMat, 0, padH, 0, padW, cv::BORDER_CONSTANT, value);
    }
}

APP_ERROR DeeplabV3plus::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * imageMat.elemSize();
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {
        static_cast<uint32_t>(imageMat.rows),
        static_cast<uint32_t>(imageMat.cols),
        static_cast<uint32_t>(imageMat.channels())};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR DeeplabV3plus::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DeeplabV3plus::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    ReadImage(imgPath, imageMat);

    MxBase::ResizedImageInfo resizedImageInfo;
    ResizeImage(imageMat, imageMat, resizedImageInfo);
    Normalize(imageMat, imageMat);
    Padding(imageMat, imageMat);

    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::SemanticSegInfo> semanticSegInfos = {};
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {resizedImageInfo};
    ret = post_->Process(outputs, semanticSegInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = imgPath;
    size_t pos = resultPath.find_last_of(".");
    resultPath.replace(resultPath.begin() + pos, resultPath.end(), "_infer.png");
    SaveResultToImage(semanticSegInfos[0], resultPath);
    return APP_ERR_OK;
}

void DeeplabV3plus::SaveResultToImage(const MxBase::SemanticSegInfo &segInfo, const std::string &filePath) {
    cv::Mat imageMat(segInfo.pixels.size(), segInfo.pixels[0].size(), CV_8UC1);
    for (size_t x = 0; x < segInfo.pixels.size(); ++x) {
        for (size_t y = 0; y < segInfo.pixels[x].size(); ++y) {
            uint8_t gray = segInfo.pixels[x][y];
            imageMat.at<uchar>(x, y) = gray;
        }
    }

    cv::Mat imageGrayC3 = cv::Mat::zeros(imageMat.rows, imageMat.cols, CV_8UC3);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < 3; i++) {
        planes.push_back(imageMat);
    }
    cv::merge(planes, imageGrayC3);
    uchar rgbColorMap[256*3] = {
        0, 0, 0,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        0, 0, 128,
        128, 0, 128,
        0, 128, 128,
        128, 128, 128,
        64, 0, 0,
        192, 0, 0,
        64, 128, 0,
        192, 128, 0,
        64, 0, 128,
        192, 0, 128,
        64, 128, 128,
        192, 128, 128,
        0, 64, 0,
        128, 64, 0,
        0, 192, 0,
        128, 192, 0,
        0, 64, 128,
    };
    cv::Mat lut(1, 256, CV_8UC3, rgbColorMap);

    cv::Mat imageColor;
    cv::LUT(imageGrayC3, lut, imageColor);
    cv::cvtColor(imageColor, imageColor, cv::COLOR_RGB2BGR);
    cv::imwrite(filePath, imageColor);
}
