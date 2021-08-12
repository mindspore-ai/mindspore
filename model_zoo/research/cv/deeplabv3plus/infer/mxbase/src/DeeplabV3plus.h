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

#ifndef DEEPLABV3PLUS_H
#define DEEPLABV3PLUS_H
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "SegmentPostProcessors/Deeplabv3Post.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    uint32_t classNum;
    uint32_t modelType;
    bool checkModel;
    uint32_t frameworkType;
};

class DeeplabV3plus {
 public:
     APP_ERROR Init(const InitParam &initParam);
     void DeInit();
     void ReadImage(const std::string &imgPath, cv::Mat &imageMat);
     void ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
         MxBase::ResizedImageInfo &resizedImageInfo);
     void Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
     void Padding(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
     APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
     APP_ERROR Process(const std::string &imgPath);
     void SaveResultToImage(const MxBase::SemanticSegInfo &segInfo, const std::string &filePath);

 private:
     std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     std::shared_ptr<MxBase::Deeplabv3Post> post_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
};

#endif
