/*
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
 *  ============================================================================
 */

#ifndef MXVISION_SSDMOBILENETFPN_MINDSPOREPOST_H
#define MXVISION_SSDMOBILENETFPN_MINDSPOREPOST_H

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {
class SsdMobilenetFpn_MindsporePost : public ObjectPostProcessBase {
 public:
    SsdMobilenetFpn_MindsporePost() = default;

    ~SsdMobilenetFpn_MindsporePost() = default;

    SsdMobilenetFpn_MindsporePost(const SsdMobilenetFpn_MindsporePost &other) = default;

    SsdMobilenetFpn_MindsporePost &operator=(const SsdMobilenetFpn_MindsporePost &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors,
                      std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ImagePreProcessInfo> &imagePreProcessInfos = {}) override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

    uint64_t GetCurrentVersion() override {
        return CURRENT_VERSION;
    }

 private:
    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                               const std::vector<ResizedImageInfo> &resizedImageInfos);
    void NonMaxSuppression(std::vector<MxBase::DetectBox>& detBoxes,
                           TensorBase &bboxTensor, TensorBase &confidenceTensor, int stride,
                           const ResizedImageInfo &imgInfo,
                           uint32_t batchNum, uint32_t batchSize);
    void NmsSort(std::vector<DetectBox>& detBoxes, float iouThresh, IOUMethod method);
    void FilterByIou(std::vector<DetectBox> dets, std::vector<DetectBox>& sortBoxes, float iouThresh, IOUMethod method);
    float CalcIou(DetectBox boxA, DetectBox boxB, IOUMethod method);

    const int CURRENT_VERSION = 2000001;
    const float DEFAULT_IOU_THRESH = 0.6;
    const int DEFAULT_OBJECT_BBOX_TENSOR = 0;
    const int DEFAULT_OBJECT_CONFIDENCE_TENSOR = 1;
    const int DEFAULT_MAX_BBOX_PER_CLASS = 100;

    float iouThresh_ = DEFAULT_IOU_THRESH;
    int objectBboxTensor_ = DEFAULT_OBJECT_BBOX_TENSOR;
    int objectConfidenceTensor_ = DEFAULT_OBJECT_CONFIDENCE_TENSOR;
    int maxBboxPerClass_ = DEFAULT_MAX_BBOX_PER_CLASS;
};

#ifdef ENABLE_POST_PROCESS_INSTANCE
extern "C" {
std::shared_ptr<MxBase::SsdMobilenetFpn_MindsporePost> GetObjectInstance();
}
#endif
}  // namespace MxBase
#endif  // MXVISION_SSDMOBILENETFPN_MINDSPOREPOST_H
