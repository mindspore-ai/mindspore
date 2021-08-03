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

#include "Yolov4TinyMindsporePost.h"
#include "MxBase/Log/Log.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
const int SCALE = 32;
const int BIASESDIM = 2;
const int OFFSETWIDTH = 2;
const int OFFSETHEIGHT = 3;
const int OFFSETBIASES = 1;
const int OFFSETOBJECTNESS = 1;

const int NHWC_HEIGHTINDEX = 1;
const int NHWC_WIDTHINDEX = 2;
const int NCHW_HEIGHTINDEX = 2;
const int NCHW_WIDTHINDEX = 3;
const int YOLO_INFO_DIM = 5;

auto uint8Deleter = [] (uint8_t* p) { };
}

namespace MxBase {
Yolov4TinyPostProcess& Yolov4TinyPostProcess::operator=(const Yolov4TinyPostProcess &other)
{
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    objectnessThresh_ = other.objectnessThresh_; // Threshold of objectness value
    iouThresh_ = other.iouThresh_;
    anchorDim_ = other.anchorDim_;
    biasesNum_ = other.biasesNum_;
    yoloType_ = other.yoloType_;
    modelType_ = other.modelType_;
    yoloType_ = other.yoloType_;
    inputType_ = other.inputType_;
    biases_ = other.biases_;
    return *this;
}

APP_ERROR Yolov4TinyPostProcess::Init(const std::map<std::string, std::shared_ptr<void>>& postConfig)
{
    LogDebug << "Start to Init Yolov4TinyPostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }

    configData_.GetFileValue<int>("BIASES_NUM", biasesNum_);
    std::string str;
    configData_.GetFileValue<std::string>("BIASES", str);
    configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_);
    configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
    configData_.GetFileValue<int>("YOLO_TYPE", yoloType_);
    configData_.GetFileValue<int>("MODEL_TYPE", modelType_);
    configData_.GetFileValue<int>("YOLO_VERSION", yoloVersion_);
    configData_.GetFileValue<int>("INPUT_TYPE", inputType_);
    configData_.GetFileValue<int>("ANCHOR_DIM", anchorDim_);
    ret = GetBiases(str);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to get biases.";
        return ret;
    }
    LogDebug << "End to Init Yolov4TinyPostProcess.";
    return APP_ERR_OK;
}

APP_ERROR Yolov4TinyPostProcess::DeInit()
{
    return APP_ERR_OK;
}

bool Yolov4TinyPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const
{   
    if (tensors.size() != (size_t)yoloType_) {
        LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to yoloType_("
                 << yoloType_ << ")";
        return false;
    }
    if (yoloVersion_ == YOLOV4_VERSION) {
        for (size_t i = 0; i < tensors.size(); i++) {
            auto shape = tensors[i].GetShape();
            if (shape.size() < VECTOR_FIFTH_INDEX) {
                LogError << "dimensions of tensor [" << i << "] is less than " << VECTOR_FIFTH_INDEX << ".";
                return false;
            }
            uint32_t channelNumber = 1;
            int startIndex = modelType_ ? VECTOR_SECOND_INDEX : VECTOR_FOURTH_INDEX;
            int endIndex = modelType_ ? (shape.size() - VECTOR_THIRD_INDEX) : shape.size();
            for (int i = startIndex; i < endIndex; i++) {
                channelNumber *= shape[i];
            }
            if (channelNumber != anchorDim_ * (classNum_ + YOLO_INFO_DIM)) {
                LogError << "channelNumber(" << channelNumber << ") != anchorDim_ * (classNum_ + 5).";
                return false;
            }
        }
        return true;
    } else {
        return true;
    }
}

void Yolov4TinyPostProcess::ObjectDetectionOutput(const std::vector<TensorBase>& tensors,
                                              std::vector<std::vector<ObjectInfo>>& objectInfos,
                                              const std::vector<ResizedImageInfo>& resizedImageInfos)
{
    LogDebug << "Yolov4TinyPostProcess start to write results.";
    if (tensors.size() == 0) {
        return;
    }
    auto shape = tensors[0].GetShape();
    if (shape.size() == 0) {
        return;
    }
    uint32_t batchSize = shape[0];
    for (uint32_t i = 0; i < batchSize; i++) {
        std::vector<std::shared_ptr<void>> featLayerData = {};
        std::vector<std::vector<size_t>> featLayerShapes = {};
        for (uint32_t j = 0; j < tensors.size(); j++) {
            auto dataPtr = (uint8_t *) tensors[j].GetBuffer() + i * tensors[j].GetByteSize() / batchSize;
            std::shared_ptr<void> tmpPointer;
            tmpPointer.reset(dataPtr, uint8Deleter);
            featLayerData.push_back(tmpPointer);
            shape = tensors[j].GetShape();
            std::vector<size_t> featLayerShape = {};
            for (auto s : shape) {
                featLayerShape.push_back((size_t)s);
            }
            featLayerShapes.push_back(featLayerShape);
        }
        std::vector<ObjectInfo> objectInfo;
        GenerateBbox(featLayerData, objectInfo, featLayerShapes, resizedImageInfos[i].widthResize,
            resizedImageInfos[i].heightResize);
        //LogInfo << "objectInfo.size before nms::::::::::" <<objectInfo.size() << ", iouThresh_:" << iouThresh_;
        MxBase::NmsSort(objectInfo, iouThresh_);
        //LogInfo << "objectInfo.size after nms::::::::::" <<objectInfo.size();
        objectInfos.push_back(objectInfo);
    }
    LogDebug << "Yolov4TinyPostProcess write results successed.";
}

APP_ERROR Yolov4TinyPostProcess::Process(const std::vector<TensorBase> &tensors,
                                     std::vector<std::vector<ObjectInfo>> &objectInfos,
                                     const std::vector<ResizedImageInfo> &resizedImageInfos,
                                     const std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogDebug << "Start to Process Yolov4TinyPostProcess.";
    APP_ERROR ret = APP_ERR_OK;
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }

    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);

    for (uint32_t i = 0; i < resizedImageInfos.size(); i++) {
        CoordinatesReduction(i, resizedImageInfos[i], objectInfos[i]);
    }
    LogObjectInfos(objectInfos);
    LogDebug << "End to Process Yolov4TinyPostProcess.";
    return APP_ERR_OK;
}

/*
 * @description: Compare the confidences between 2 classes and get the larger one
 */
void Yolov4TinyPostProcess::CompareProb(int& classID, float& maxProb, float classProb, int classNum)
{
    if (classProb > maxProb) {
        maxProb = classProb;
        classID = classNum;
    }
}

void Yolov4TinyPostProcess::SelectClassNHWC(std::shared_ptr<void> netout, NetInfo info,
                                          std::vector<MxBase::ObjectInfo>& detBoxes, int stride, OutputLayer layer)
{
    const int offsetY = 1;
    for (int j = 0; j < stride; ++j) {
        for (int k = 0; k < info.anchorDim; ++k) {
            int bIdx = (info.bboxDim + 1 + info.classNum) * info.anchorDim * j +
                       k * (info.bboxDim + 1 + info.classNum);
            int oIdx = bIdx + info.bboxDim; // objectness index
            // check obj
            float objectness = static_cast<float *>(netout.get())[oIdx];
            if (objectness <= objectnessThresh_) {
                continue;
            }
            int classID = -1;
            float maxProb = scoreThresh_;
            float classProb;
            // Compare the confidence of the 3 anchors, select the largest one
            for (int c = 0; c < info.classNum; ++c) {
                classProb = static_cast<float *>(netout.get())[bIdx +
                    (info.bboxDim + OFFSETOBJECTNESS + c)] * objectness;
                CompareProb(classID, maxProb, classProb, c);
            }
            //LogInfo << "maxProb:" << maxProb << ", objectness:" << objectness;
            if (classID < 0) continue;
            MxBase::ObjectInfo det;
            int row = j / layer.width;
            int col = j % layer.width;
            float x = static_cast<float *>(netout.get())[bIdx];
            float y = static_cast<float *>(netout.get())[bIdx + offsetY];
            float width = static_cast<float *>(netout.get())[bIdx + OFFSETWIDTH];
            float height = static_cast<float *>(netout.get())[bIdx + OFFSETHEIGHT];
            det.x0 = std::max(0.0f, x - width / COORDINATE_PARAM);
            det.x1 = std::min(1.0f, x + width / COORDINATE_PARAM);
            det.y0 = std::max(0.0f, y - height / COORDINATE_PARAM);
            det.y1 = std::min(1.0f, y + height / COORDINATE_PARAM);
            //LogInfo << "x0,y0,x1,y1: " << det.x0 << "," << det.y0 << "," << det.x1 << "," << det.y1;
            det.classId = classID;
            det.className = configData_.GetClassName(classID);
            det.confidence = maxProb;
            if (det.confidence < separateScoreThresh_[classID]) continue;
            detBoxes.emplace_back(det);
        }
    }
}

/*
 * @description: According to the yolo layer structure, encapsulate the anchor box data of each feature into detBoxes
 * @param featLayerData  Vector of 3 output feature data
 * @param info  Yolo layer info which contains anchors dim, bbox dim, class number, net width, net height and
                3 outputlayer(eg. 13*13, 26*26, 52*52)
 * @param detBoxes  ObjectInfo vector where all ObjectInfoes's confidences are greater than threshold
 */
void Yolov4TinyPostProcess::GenerateBbox(std::vector<std::shared_ptr<void>> featLayerData,
                                     std::vector<MxBase::ObjectInfo> &detBoxes,
                                     const std::vector<std::vector<size_t>>& featLayerShapes, const int netWidth,
                                     const int netHeight)
{
    NetInfo netInfo;
    netInfo.anchorDim = anchorDim_;
    netInfo.bboxDim = BOX_DIM;
    netInfo.classNum = classNum_;
    netInfo.netWidth = netWidth;
    netInfo.netHeight = netHeight;
    for (int i = 0; i < yoloType_; ++i) {
        int widthIndex_ = modelType_ ? NCHW_WIDTHINDEX : NHWC_WIDTHINDEX;
        int heightIndex_ = modelType_ ? NCHW_HEIGHTINDEX : NHWC_HEIGHTINDEX;
        OutputLayer layer = {featLayerShapes[i][widthIndex_], featLayerShapes[i][heightIndex_]};
        int logOrder = log(featLayerShapes[i][widthIndex_] * SCALE / netWidth) / log(BIASESDIM);
        int startIdx = (yoloType_ - 1 - logOrder) * netInfo.anchorDim * BIASESDIM;
        int endIdx = startIdx + netInfo.anchorDim * BIASESDIM;
        int idx = 0;
        for (int j = startIdx; j < endIdx; ++j) {
            layer.anchors[idx++] = biases_[j];
        }
        // 13*13 26*26 52*52
        int stride = layer.width * layer.height;
        std::shared_ptr<void> netout = featLayerData[i];
        SelectClassNHWC(netout, netInfo, detBoxes, stride, layer);
    }
}

APP_ERROR Yolov4TinyPostProcess::GetBiases(std::string& strBiases)
{
    if (biasesNum_ <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "Failed to get biasesNum (" << biasesNum_ << ").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.clear();
    int i = 0;
    int num = strBiases.find(",");
    while (num >= 0 && i < biasesNum_) {
        std::string tmp = strBiases.substr(0, num);
        num++;
        strBiases = strBiases.substr(num, strBiases.size());
        biases_.push_back(stof(tmp));
        i++;
        num = strBiases.find(",");
    }
    if (i != biasesNum_ - 1 || strBiases.size() <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "biasesNum (" << biasesNum_
                 << ") is not equal to total number of biases (" << strBiases <<").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.push_back(stof(strBiases));
    return APP_ERR_OK;
}

#ifndef ENABLE_POST_PROCESS_INSTANCE
extern "C" {
std::shared_ptr<MxBase::Yolov4TinyPostProcess> GetObjectInstance()
{
    LogInfo << "Begin to get Yolov4TinyPostProcess instance.";
    auto instance = std::make_shared<Yolov4TinyPostProcess>();
    LogInfo << "End to get Yolov4TinyPostProcess instance.";
    return instance;
}
}
#endif
}

