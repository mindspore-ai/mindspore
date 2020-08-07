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

#include "tools/converter/legacy_optimizer/fusion/batchnorm_fold_fusion_pass.h"
#include <cfloat>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utils/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/op_utils.h"

namespace mindspore {
namespace lite {
#define kBatchNormFoldFusionPathLen6 6
#define kBatchNormFoldFusionPathLen7 7

STATUS BatchNormFoldFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS BatchNormFoldFusionPass::DefinePattern() {
  // with preNode
  {
    auto inputOp = std::make_shared<PatternOp>();
    inputOp->id = inputOpName;
    inputOp->types = {schema::PrimitiveType_NONE};
    inputOp->isPlaceHold = true;

    auto convOp1 = std::make_shared<PatternOp>();
    convOp1->id = convPatternOpName1;
    convOp1->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
    convOp1->left = inputOp;

    auto bnFoldOp = std::make_shared<PatternOp>();
    bnFoldOp->id = bnFoldOpName;
    bnFoldOp->types = {schema::PrimitiveType_BatchNormFold};
    bnFoldOp->left = convOp1;

    auto mulFoldOp = std::make_shared<PatternOp>();
    mulFoldOp->id = mulFoldOpName;
    mulFoldOp->types = {schema::PrimitiveType_MulFold};
    mulFoldOp->left = bnFoldOp;

    auto fakeQuantOp = std::make_shared<PatternOp>();
    fakeQuantOp->id = fakeQuantOpName;
    fakeQuantOp->types = {schema::PrimitiveType_FakeQuantWithMinMax};
    fakeQuantOp->left = mulFoldOp;

    auto convOp2 = std::make_shared<PatternOp>();
    convOp2->id = convPatternOpName2;
    convOp2->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
    convOp2->left = fakeQuantOp;
    convOp2->right = inputOp;

    auto addFoldOp = std::make_shared<PatternOp>();
    addFoldOp->id = addFoldOpName;
    addFoldOp->types = {schema::PrimitiveType_AddFold};
    addFoldOp->left = convOp2;
    addFoldOp->right = bnFoldOp;

    std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern(withPrePatternName));
    if (fusionPattern == nullptr) {
      MS_LOG(ERROR) << "new fusionPattern failed";
      return RET_ERROR;
    }
    fusionPattern->AddPatternOp(inputOp);
    fusionPattern->AddPatternOp(convOp1);
    fusionPattern->AddPatternOp(bnFoldOp);
    fusionPattern->AddPatternOp(mulFoldOp);
    fusionPattern->AddPatternOp(fakeQuantOp);
    fusionPattern->AddPatternOp(convOp2);
    fusionPattern->AddPatternOp(addFoldOp);
    fusionPattern->Finish();

    this->patterns.emplace_back(fusionPattern.release());
  }
  // no preNode
  {
    auto convOp1 = std::make_shared<PatternOp>();
    convOp1->id = convPatternOpName1;
    convOp1->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};

    auto bnFoldOp = std::make_shared<PatternOp>();
    bnFoldOp->id = bnFoldOpName;
    bnFoldOp->types = {schema::PrimitiveType_BatchNormFold};
    bnFoldOp->left = convOp1;

    auto mulFoldOp = std::make_shared<PatternOp>();
    mulFoldOp->id = mulFoldOpName;
    mulFoldOp->types = {schema::PrimitiveType_MulFold};
    mulFoldOp->left = bnFoldOp;

    auto fakeQuantOp = std::make_shared<PatternOp>();
    fakeQuantOp->id = fakeQuantOpName;
    fakeQuantOp->types = {schema::PrimitiveType_FakeQuantWithMinMax};
    fakeQuantOp->left = mulFoldOp;

    auto convOp2 = std::make_shared<PatternOp>();
    convOp2->id = convPatternOpName2;
    convOp2->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
    convOp2->left = fakeQuantOp;

    auto addFoldOp = std::make_shared<PatternOp>();
    addFoldOp->id = addFoldOpName;
    addFoldOp->types = {schema::PrimitiveType_AddFold};
    addFoldOp->left = convOp2;
    addFoldOp->right = bnFoldOp;

    std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern(noPrePatternName));
    if (fusionPattern == nullptr) {
      MS_LOG(ERROR) << "new fusionPattern failed";
      return RET_ERROR;
    }
    fusionPattern->AddPatternOp(convOp1);
    fusionPattern->AddPatternOp(bnFoldOp);
    fusionPattern->AddPatternOp(mulFoldOp);
    fusionPattern->AddPatternOp(fakeQuantOp);
    fusionPattern->AddPatternOp(convOp2);
    fusionPattern->AddPatternOp(addFoldOp);
    fusionPattern->Finish();

    this->patterns.emplace_back(fusionPattern.release());
  }
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::DoFusion(MetaGraphT *graph, const std::string &patternName,
                                         std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (patternName == withPrePatternName) {
    if (matchedPath.size() != kBatchNormFoldFusionPathLen7) {
      MS_LOG(ERROR) << "BatchNormFold-Fusion should have seven NodeIndex in matchedPair";
      return RET_PARAM_INVALID;
    }
  } else if (patternName == noPrePatternName) {
    if (matchedPath.size() != kBatchNormFoldFusionPathLen6) {
      MS_LOG(ERROR) << "BatchNormFold-Fusion should have six NodeIndex in matchedPair";
      return RET_PARAM_INVALID;
    }
  }

  auto status = FindNodes(graph, matchedPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FindNodes failed: " << status;
    return status;
  }
  status = CheckPath(graph, matchedPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CheckPath failed: " << status;
    return status;
  }
  status = FindTensors();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FindTensors failed: " << status;
    return status;
  }
  status = GenNewWeightTensor();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GenNewWeightTensor failed: " << status;
    return status;
  }
  status = GenNewBiasTensor();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GenNewBiasTensor failed: " << status;
    return status;
  }
  status = IsolateNodes(graph, matchedPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateNodes failed: " << status;
    return status;
  }
  UpdateConvWeights();
  status = DeleteConstTensors();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DeleteConstTensors failed: " << status;
    return status;
  }
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::FindNodes(MetaGraphT *graph,
                                          const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  auto preConvPath = matchedPath.at(convPatternOpName1);
  auto bnFoldPath = matchedPath.at(bnFoldOpName);
  auto mulFoldPath = matchedPath.at(mulFoldOpName);
  auto fakeQuantPath = matchedPath.at(fakeQuantOpName);
  auto convPath = matchedPath.at(convPatternOpName2);
  auto addFoldPath = matchedPath.at(addFoldOpName);
  MS_ASSERT(preConvPath != nullptr);
  MS_ASSERT(bnFoldPath != nullptr);
  MS_ASSERT(mulFoldPath != nullptr);
  MS_ASSERT(fakeQuantPath != nullptr);
  MS_ASSERT(convPath != nullptr);
  MS_ASSERT(addFoldPath != nullptr);
  if (preConvPath->subGraphIdx != bnFoldPath->subGraphIdx || preConvPath->subGraphIdx != mulFoldPath->subGraphIdx ||
      preConvPath->subGraphIdx != fakeQuantPath->subGraphIdx || preConvPath->subGraphIdx != convPath->subGraphIdx ||
      preConvPath->subGraphIdx != addFoldPath->subGraphIdx) {
    MS_LOG(ERROR) << "matched nodes should from same subGraph";
    return RET_ERROR;
  }
  MS_ASSERT(graph->nodes.size() > preConvPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > bnFoldPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > mulFoldPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > fakeQuantPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > convPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > addFoldPath->nodeIdx);
  preConv = graph->nodes.at(preConvPath->nodeIdx).get();
  bnFold = graph->nodes.at(bnFoldPath->nodeIdx).get();
  mulFold = graph->nodes.at(mulFoldPath->nodeIdx).get();
  fakeNode = graph->nodes.at(fakeQuantPath->nodeIdx).get();
  convNode = graph->nodes.at(convPath->nodeIdx).get();
  addFold = graph->nodes.at(addFoldPath->nodeIdx).get();
  MS_ASSERT(preConv != nullptr);
  MS_ASSERT(bnFold != nullptr);
  MS_ASSERT(mulFold != nullptr);
  MS_ASSERT(fakeNode != nullptr);
  MS_ASSERT(convNode != nullptr);
  MS_ASSERT(addFold != nullptr);
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::FindTensors() {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnFold != nullptr);
  MS_ASSERT(addFold != nullptr);
  if (bnFold->inputIndex.size() != 4) {
    MS_LOG(ERROR) << "BatchNormFold node should have 4 inputTensor, got " << bnFold->inputIndex.size()
                  << " input tensors";
    return RET_ERROR;
  }
  if (addFold->inputIndex.size() != 5) {
    MS_LOG(ERROR) << "AddFold node should have 5 inputTensor, got " << addFold->inputIndex.size() << " input tensors";
    return RET_ERROR;
  }
  MS_ASSERT(graph->allTensors.size() > bnFold->inputIndex.at(1));
  muTensor = graph->allTensors.at(bnFold->inputIndex.at(1)).get();
  MS_ASSERT(muTensor != nullptr);
  MS_ASSERT(graph->allTensors.size() > bnFold->inputIndex.at(2));
  sigmaTensor = graph->allTensors.at(bnFold->inputIndex.at(2)).get();
  MS_ASSERT(sigmaTensor != nullptr);
  MS_ASSERT(graph->allTensors.size() > addFold->inputIndex.at(1));
  betaTensor = graph->allTensors.at(addFold->inputIndex.at(1)).get();
  MS_ASSERT(betaTensor != nullptr);
  MS_ASSERT(graph->allTensors.size() > addFold->inputIndex.at(2));
  gammaTensor = graph->allTensors.at(addFold->inputIndex.at(2)).get();
  MS_ASSERT(gammaTensor != nullptr);

  if (betaTensor->dims.size() != 1) {
    MS_LOG(ERROR) << "ConstTensor should have only one dim, got " << betaTensor->dims.size();
    return RET_ERROR;
  }
  if (betaTensor->dims != gammaTensor->dims || betaTensor->dims != sigmaTensor->dims ||
      betaTensor->dims != muTensor->dims) {
    MS_LOG(ERROR) << "All ConstTensor should have same dims";
    return RET_ERROR;
  }
  channelOut = betaTensor->dims.front();

  MS_ASSERT(mulFold != nullptr);
  if (mulFold->inputIndex.size() != 3) {
    MS_LOG(ERROR) << "MulFold node should have 3 outputTensor, got " << addFold->inputIndex.size() << " output tensors";
    return RET_ERROR;
  }
  MS_ASSERT(graph->allTensors.size() > mulFold->inputIndex.front());
  oldWeightTensor = graph->allTensors.at(mulFold->inputIndex.front()).get();
  MS_ASSERT(oldWeightTensor != nullptr);
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::CheckPath(MetaGraphT *graph,
                                          const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(preConv != nullptr);
  MS_ASSERT(convNode != nullptr);
  MS_ASSERT(mulFold != nullptr);
  MS_ASSERT(preConv->inputIndex.size() == 2);
  MS_ASSERT(convNode->inputIndex.size() == 2);
  MS_ASSERT(mulFold->inputIndex.size() == 3);
  MS_ASSERT(preConv->inputIndex.front() == convNode->inputIndex.front());
  MS_ASSERT(preConv->inputIndex.at(1) == mulFold->inputIndex.front());
  // todo
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::GenNewWeightTensor() {
  MS_ASSERT(oldWeightTensor != nullptr);
  MS_ASSERT(oldWeightTensor->dataType == DataType_DT_FLOAT);
  MS_ASSERT(oldWeightTensor->refCount == schema::NodeType_ValueNode);
  auto weightShape = oldWeightTensor->dims;
  if (weightShape.size() != 4) {
    MS_LOG(ERROR) << "shape of weight should be 4 dims, got " << weightShape.size() << " dims";
    return RET_ERROR;
  }
  if (weightShape.front() != channelOut) {
    MS_LOG(ERROR) << "weight should be in KCHW format, and outputChannel should be " << channelOut;
    return RET_ERROR;
  }
  auto weightShapeSize = GetShapeSize(*oldWeightTensor);
  newWeightTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newWeightTensor == nullptr) {
    MS_LOG(ERROR) << "new weightTensor failed";
    return RET_ERROR;
  }
  newWeightTensor->dataType = oldWeightTensor->dataType;
  newWeightTensor->format = oldWeightTensor->format;
  newWeightTensor->refCount = schema::NodeType_ValueNode;
  newWeightTensor->dims = weightShape;
  newWeightTensor->data.resize(weightShapeSize * sizeof(float));
  void *oldWeightData = oldWeightTensor->data.data();
  auto castedOldWeightData = static_cast<float *>(oldWeightData);
  void *newWeightData = newWeightTensor->data.data();
  auto castedNewWeightData = static_cast<float *>(newWeightData);
  MS_ASSERT(gammaTensor->dataType == DataType_DT_FLOAT);
  void *gammaData = gammaTensor->data.data();
  auto *castedGammaData = static_cast<float *>(gammaData);
  MS_ASSERT(muTensor->dataType == DataType_DT_FLOAT);
  void *miData = muTensor->data.data();
  auto *castedMiData = static_cast<float *>(miData);
  size_t stride = weightShapeSize / channelOut;
  for (size_t i = 0; i < channelOut; i++) {
    for (size_t j = 0; j < stride; j++) {
      castedNewWeightData[i * stride + j] = castedOldWeightData[i * stride + j] * castedGammaData[i] / castedMiData[i];
    }
  }
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::GenNewBiasTensor() {  // bias has no quant
  std::vector<int32_t> biasShape = {channelOut};
  newBiasTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newBiasTensor == nullptr) {
    MS_LOG(ERROR) << "new BiasTensor failed";
    return RET_ERROR;
  }
  newBiasTensor->dataType = 0;  // todo is float
  newBiasTensor->format = Format_NUM_OF_FORMAT;
  newBiasTensor->refCount = schema::NodeType_ValueNode;
  newBiasTensor->dims = biasShape;
  newBiasTensor->data.resize(channelOut * sizeof(float));
  void *newBiasData = newBiasTensor->data.data();
  auto castedNewBiasData = static_cast<float *>(newBiasData);
  MS_ASSERT(betaTensor->dataType == DataType_DT_FLOAT);
  void *betaData = betaTensor->data.data();
  auto *castedBetaData = static_cast<float *>(betaData);
  MS_ASSERT(gammaTensor->dataType == DataType_DT_FLOAT);
  void *gammaData = gammaTensor->data.data();
  auto *castedGammaData = static_cast<float *>(gammaData);
  MS_ASSERT(muTensor->dataType == DataType_DT_FLOAT);
  void *miData = muTensor->data.data();
  auto *castedMiData = static_cast<float *>(miData);
  MS_ASSERT(sigmaTensor->dataType == DataType_DT_FLOAT);
  void *sigmaData = sigmaTensor->data.data();
  auto *castedSigmaData = static_cast<float *>(sigmaData);
  for (size_t i = 0; i < channelOut; i++) {
    castedNewBiasData[i] = castedBetaData[i] - castedGammaData[i] * castedMiData[i] / castedSigmaData[i];
  }
  return RET_OK;
}

STATUS BatchNormFoldFusionPass::IsolateNodes(
  MetaGraphT *graph, const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  auto preConvPath = matchedPath.at(convPatternOpName1);
  auto bnFoldPath = matchedPath.at(bnFoldOpName);
  auto mulFoldPath = matchedPath.at(mulFoldOpName);
  auto fakeQuantPath = matchedPath.at(fakeQuantOpName);
  auto convPath = matchedPath.at(convPatternOpName2);
  auto addFoldPath = matchedPath.at(addFoldOpName);
  MS_ASSERT(preConvPath != nullptr);
  MS_ASSERT(bnFoldPath != nullptr);
  MS_ASSERT(mulFoldPath != nullptr);
  MS_ASSERT(fakeQuantPath != nullptr);
  MS_ASSERT(convPath != nullptr);
  MS_ASSERT(addFoldPath != nullptr);
  auto status = IsolateOneWayNode(graph, preConvPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode " << preConv->name.c_str() << " failed, error: " << status;
    return status;
  }
  std::vector<uint32_t> toDeleteTensorIdxes;
  toDeleteTensorIdxes.emplace_back(bnFold->inputIndex.at(3));
  toDeleteTensorIdxes.insert(toDeleteTensorIdxes.end(), bnFold->outputIndex.begin(), bnFold->outputIndex.end());
  status = RemoveTensor(graph, toDeleteTensorIdxes, true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Remove Tensors of BnFold " << bnFold->name.c_str() << " failed, error: " << status;
    return RET_ERROR;
  }
  status = IsolateOneWayNode(graph, bnFoldPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode " << bnFold->name.c_str() << " failed, error: " << status;
    return status;
  }
  status = IsolateOneWayNode(graph, mulFoldPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode " << mulFold->name.c_str() << " failed, error: " << status;
    return status;
  }
  status = IsolateOneWayNode(graph, addFoldPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode " << addFold->name.c_str() << " failed, error: " << status;
    return status;
  }
  return RET_OK;
}

void BatchNormFoldFusionPass::UpdateConvWeights() {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(convNode != nullptr);
  MS_ASSERT(newWeightTensor != nullptr);
  MS_ASSERT(newBiasTensor != nullptr);
  MS_ASSERT(graph->allTensors.size() > fakeNode->inputIndex.at(0));
  graph->allTensors.at(fakeNode->inputIndex.at(0)).reset();
  graph->allTensors.at(fakeNode->inputIndex.at(0)) = std::move(this->newWeightTensor);
  graph->allTensors.emplace_back(std::move(this->newBiasTensor));
  convNode->inputIndex.emplace_back(graph->allTensors.size() - 1);
  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    convNode->primitive->value.AsConv2D()->hasBias = true;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    convNode->primitive->value.AsDepthwiseConv2D()->hasBias = true;
  } else {
    MS_ASSERT(false);
  }

  this->oldWeightTensor = nullptr;
  this->newWeightTensor = nullptr;
  this->newBiasTensor = nullptr;
}

STATUS BatchNormFoldFusionPass::DeleteConstTensors() {
  MS_ASSERT(graph != nullptr);
  bool muFind = false;
  bool sigmaFind = false;
  bool betaFind = false;
  bool gammaFind = false;
  std::vector<uint32_t> toDeleteTensorIdxes;
  for (size_t i = 0; i < graph->allTensors.size(); i++) {
    auto &tensor = graph->allTensors.at(i);
    if (tensor.get() == muTensor) {
      toDeleteTensorIdxes.emplace_back(i);
      muFind = true;
      this->muTensor = nullptr;
    }
    if (tensor.get() == sigmaTensor) {
      toDeleteTensorIdxes.emplace_back(i);
      sigmaFind = true;
      this->sigmaTensor = nullptr;
    }
    if (tensor.get() == gammaTensor) {
      toDeleteTensorIdxes.emplace_back(i);
      gammaFind = true;
      this->gammaTensor = nullptr;
    }
    if (tensor.get() == betaTensor) {
      toDeleteTensorIdxes.emplace_back(i);
      betaFind = true;
      this->betaTensor = nullptr;
    }
  }
  if (!muFind || !sigmaFind || !betaFind || !gammaFind) {
    MS_LOG(ERROR) << "Can not find muTensor or sigmaTensor or betaTensor or gammaTensor in graph";
    return RET_ERROR;
  }
  auto status = RemoveTensor(graph, toDeleteTensorIdxes);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Remove ConstTensors failed" << bnFold->name.c_str();
    return RET_ERROR;
  }
  return RET_OK;
}

BatchNormFoldFusionPass::~BatchNormFoldFusionPass() {
  if (newWeightTensor == nullptr) {
    newWeightTensor.reset();
    newWeightTensor = nullptr;
  }
  if (newBiasTensor == nullptr) {
    newBiasTensor.reset();
    newBiasTensor = nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore

