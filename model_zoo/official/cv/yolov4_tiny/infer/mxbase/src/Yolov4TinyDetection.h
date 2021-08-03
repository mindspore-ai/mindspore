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

#ifndef MXBASE_YOLOV4TINYDETECTIONOPENCV_H
#define MXBASE_YOLOV4TINYDETECTIONOPENCV_H

#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "PostProcess/Yolov4TinyMindsporePost.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    bool checkTensor;
    std::string modelPath;
    uint32_t classNum;
    uint32_t biasesNum;
    std::string biases;
    std::string objectnessThresh;
    std::string iouThresh;
    std::string scoreThresh;
    uint32_t yoloType;
    uint32_t modelType;
    uint32_t inputType;
    uint32_t anchorDim;
};

class Yolov4TinyDetectionOpencv {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &outputs,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    APP_ERROR WriteResult(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Yolov4TinyPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    std::map<int, std::string> labelMap_;
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
};
#endif