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

#ifndef MINDSPORE_PREDICT_BATCHNORM_FOLD_FUSION_PASS_H
#define MINDSPORE_PREDICT_BATCHNORM_FOLD_FUSION_PASS_H

#include <unordered_map>
#include <memory>
#include <string>
#include "tools/converter/legacy_optimizer/fusion/fusion_pass.h"

namespace mindspore {
namespace lite {
// input = input
// weight = SimQuantPerChannel(weight * gamma / sigma)
// bias = beta - gamma * mi / sigma
// MulFold: gamma sigma
// BatchNormFold: mi sigma
// AddFold: gamma beta mi sigma
class BatchNormFoldFusionPass : public FusionPass {
 public:
  BatchNormFoldFusionPass() = default;

  ~BatchNormFoldFusionPass() override;

  STATUS DefinePattern() override;

  STATUS DoFusion(MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(MetaGraphT *graph) override;

 protected:
  STATUS FindNodes(MetaGraphT *graph, const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath);
  STATUS CheckPath(MetaGraphT *graph, const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath);
  STATUS FindTensors();
  STATUS GenNewWeightTensor();
  STATUS GenNewBiasTensor();
  STATUS IsolateNodes(MetaGraphT *graph, const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath);
  void UpdateConvWeights();
  STATUS DeleteConstTensors();

 protected:
  MetaGraphT *graph = nullptr;
  CNodeT *preConv = nullptr;
  CNodeT *bnFold = nullptr;
  CNodeT *mulFold = nullptr;
  CNodeT *fakeNode = nullptr;
  CNodeT *convNode = nullptr;
  CNodeT *addFold = nullptr;
  TensorT *muTensor = nullptr;
  TensorT *sigmaTensor = nullptr;
  TensorT *gammaTensor = nullptr;
  TensorT *betaTensor = nullptr;
  TensorT *oldWeightTensor = nullptr;
  int32_t channelOut = 0;

  std::unique_ptr<TensorT> newWeightTensor = nullptr;
  std::unique_ptr<TensorT> newBiasTensor = nullptr;

  std::string inputOpName = "Input";
  std::string convPatternOpName1 = "Convolution1";
  std::string bnFoldOpName = "BatchNormFold";
  std::string mulFoldOpName = "MulFold";
  std::string fakeQuantOpName = "FakeQuant";
  std::string convPatternOpName2 = "Convolution2";
  std::string addFoldOpName = "AddFold";
  std::string withPrePatternName = "BNFoldFusionWithPre";
  std::string noPrePatternName = "BNFoldFusionNoPre";
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_BATCHNORM_FOLD_FUSION_PASS_H
