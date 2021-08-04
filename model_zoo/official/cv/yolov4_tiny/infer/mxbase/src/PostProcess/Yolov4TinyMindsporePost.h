/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef YOLOV4TINY_POST_PROCESS_H
#define YOLOV4TINY_POST_PROCESS_H
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

const float DEFAULT_OBJECTNESS_THRESH = 0.3;
const float DEFAULT_IOU_THRESH = 0.45;
const int DEFAULT_ANCHOR_DIM = 3;
const int DEFAULT_BIASES_NUM = 18;
const int DEFAULT_YOLO_TYPE = 3;
const int DEFAULT_YOLO_VERSION = 4;
const int YOLOV3_VERSION = 3;
const int YOLOV4_VERSION = 4;
const int YOLOV5_VERSION = 5;

struct OutputLayer {
    size_t width;
    size_t height;
    float anchors[6];
};

struct NetInfo {
    int anchorDim;
    int classNum;
    int bboxDim;
    int netWidth;
    int netHeight;
};

namespace MxBase {
class Yolov4TinyPostProcess : public ObjectPostProcessBase {
 public:
     Yolov4TinyPostProcess() = default;

     ~Yolov4TinyPostProcess() = default;

     Yolov4TinyPostProcess(const Yolov4TinyPostProcess &other) = default;

     Yolov4TinyPostProcess &operator=(const Yolov4TinyPostProcess &other);

     APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

     APP_ERROR DeInit() override;

     APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

 protected:
     bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

     void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                               const std::vector<ResizedImageInfo> &resizedImageInfos = {});

     void CompareProb(int& classID, float& maxProb, float classProb, int classNum);
     void SelectClassNHWC(std::shared_ptr<void> netout, NetInfo info, std::vector<MxBase::ObjectInfo>& detBoxes,
                         int stride, OutputLayer layer);
     void GenerateBbox(std::vector<std::shared_ptr<void>> featLayerData,
                      std::vector<MxBase::ObjectInfo> &detBoxes,
                      const std::vector<std::vector<size_t>>& featLayerShapes,
                      const int netWidth, const int netHeight);
     APP_ERROR GetBiases(std::string& strBiases);

 protected:
     float objectnessThresh_ = DEFAULT_OBJECTNESS_THRESH;  // Threshold of objectness value
     float iouThresh_ = DEFAULT_IOU_THRESH;  // Non-Maximum Suppression threshold
     int anchorDim_ = DEFAULT_ANCHOR_DIM;
     int biasesNum_ = DEFAULT_BIASES_NUM;  // anchors, generate from train data, coco dataset
     int yoloType_ = DEFAULT_YOLO_TYPE;
     int modelType_ = 0;
     int yoloVersion_ = DEFAULT_YOLO_VERSION;
     int inputType_ = 0;
     std::vector<float> biases_ = {};
};
#ifndef ENABLE_POST_PROCESS_INSTANCE
extern "C" {
std::shared_ptr<MxBase::Yolov4TinyPostProcess> GetObjectInstance();
}
#endif
}  // namespace MxBase
#endif
