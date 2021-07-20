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

#ifndef MXBASE_BERTBASE_H
#define MXBASE_BERTBASE_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;
extern uint32_t g_TP;
extern uint32_t g_FP;
extern uint32_t g_FN;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    uint32_t classNum;
};

enum DataIndex {
    INPUT_IDS = 0,
    INPUT_MASK = 1,
    TOKEN_TYPE = 2,
};

class BertNerBase {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferPath, const std::string &fileName, bool eval);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<uint32_t> *argmax);
 protected:
    APP_ERROR ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size);
    APP_ERROR ReadInputTensor(const std::string &fileName, uint32_t index, std::vector<MxBase::TensorBase> *inputs);
    APP_ERROR LoadLabels(const std::string &labelPath, std::vector<std::string> *labelMap);
    APP_ERROR ReadInputTensor(const std::string &fileName, const std::vector<uint32_t> &argmax);
    APP_ERROR WriteResult(const std::string &fileName, const std::vector<uint32_t> &argmax);
    APP_ERROR CountPredictResult(const std::string &labelFile, const std::vector<uint32_t> &argmax);
    void GetClunerLabel(const std::vector<uint32_t> &argmax,
                        std::multimap<std::string, std::vector<uint32_t>> *clunerMap);
 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    std::vector<std::string> labelMap_ = {};
    uint32_t deviceId_ = 0;
    uint32_t classNum_ = 0;
};
#endif
