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

#ifndef MXBASE_DPN_H
#define MXBASE_DPN_H
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class DPN {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *imageMat);
  APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat *dstImageMat);
  APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
  APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                          std::vector<std::vector<MxBase::ClassInfo>> *clsInfos);
  APP_ERROR Process(const std::vector<std::string> &batchImgPaths);
  // get infer time
  double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}

 private:
  APP_ERROR SaveResult(const std::vector<std::string> &batchImgPaths,
                                const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos);

 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  std::shared_ptr<MxBase::Resnet50PostProcess> post_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  // infer time
  double inferCostTimeMilliSec = 0.0;
};

#endif

