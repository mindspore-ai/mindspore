/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H
#define MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H

#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include "tools/converter/legacy_optimizer/fusion/fusion_pass.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
struct BNWeightTensors {
  TensorT *meanTensor = nullptr;
  TensorT *varianceTensor = nullptr;
  TensorT *scaleTensor = nullptr;
  TensorT *biasTensor = nullptr;
};
class BatchNormConvertScalePass : public FusionPass {
 public:
  BatchNormConvertScalePass() = default;

  ~BatchNormConvertScalePass() = default;

  STATUS DefinePattern() override;

  STATUS DoFusion(MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(MetaGraphT *graph) override;

 protected:
  STATUS GetTransParam(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath);

  // Get and check BNNode weight tensor
  STATUS GetBnWeightTensors(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath, BNWeightTensors* bnWeightTensors);

  STATUS GetBnEpsilon(MetaGraphT *graph);

  STATUS FindNodes(MetaGraphT *graph, const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath);

  STATUS GenNewScaleTensor(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath);

  STATUS ConvertBNToScale(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath);

  CNodeT *inputNode = nullptr;
  CNodeT *bnNode = nullptr;

  std::string inputOpName = "Input";
  std::string bnOpName = "BatchNorm";
  std::string bnPatternName = "BnToScaleFusion";
  uint32_t bnChannel = 0;
  float eps = 0;
  TensorT *bnMeanTensor = nullptr;
  float *transScale = nullptr;
  float *transBias = nullptr;
  std::unique_ptr<TensorT> newScaleWeightTensor = nullptr;
  std::unique_ptr<TensorT> newScaleBiasTensor = nullptr;

  OpDefCopyer ScaleOpCopyer = [](CNodeT *inOpDef) -> std::unique_ptr<CNodeT> {
    std::unique_ptr<CNodeT> newOpDef(new(std::nothrow) CNodeT);
    if (newOpDef == nullptr) {
      MS_LOG(ERROR) << "new OpDefT failed";
      return nullptr;
    }
    newOpDef->name = inOpDef->name;
    newOpDef->quantType = inOpDef->quantType;
    newOpDef->primitive = std::make_unique<schema::PrimitiveT>();
    newOpDef->primitive->value.type = schema::PrimitiveType_Scale;
    auto scaleParam = new(std::nothrow) ScaleT;
    if (scaleParam == nullptr) {
      MS_LOG(ERROR) << "new scaleParam failed";
      return nullptr;
    }
    auto inParam = inOpDef->primitive->value.AsScale();
    MS_ASSERT(inParam != nullptr);
    scaleParam->axis = inParam->axis;
    newOpDef->primitive->value.value = scaleParam;
    return std::move(newOpDef);
  };
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H
