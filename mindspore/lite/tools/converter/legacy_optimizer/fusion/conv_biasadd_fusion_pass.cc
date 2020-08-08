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

#include "tools/converter/legacy_optimizer/fusion/conv_biasadd_fusion_pass.h"
#include <cfloat>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "securec/include/securec.h"
// #include "utils/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/op_utils.h"

namespace mindspore {
namespace lite {
#define CONV_BIASADD_MATCH_PATH_LEN 2
#define BIASADD_OP_BIAS_INDEX_IN_WEIGHT 0
#define BIASADD_OP_INPUT_NUM 2
#define BIASADD_OP_CONST_TENSOR_INDEX 1

STATUS ConvBiasAddFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS ConvBiasAddFusionPass::DefinePattern() {
  auto convOp = std::make_shared<PatternOp>();
  convOp->id = kConvName;
  convOp->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_DeConv2D};
  auto baOp = std::make_shared<PatternOp>();
  baOp->id = BIASADD_NAME;
  baOp->types = {schema::PrimitiveType_BiasAdd, schema::PrimitiveType_Add};
  baOp->left = convOp;

  std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern("ConvBiasAddFusion"));
  if (fusionPattern == nullptr) {
    MS_LOG(ERROR) << "new fusionPattern failed";
    return RET_ERROR;
  }
  fusionPattern->AddPatternOp(convOp);
  fusionPattern->AddPatternOp(baOp);
  fusionPattern->Finish();

  this->patterns.emplace_back(fusionPattern.release());

  return RET_OK;
}

STATUS ConvBiasAddFusionPass::DoFusion(MetaGraphT *graph, const std::string &patternName,
                                       std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != CONV_BIASADD_MATCH_PATH_LEN) {
    MS_LOG(ERROR) << "Conv-BiasAdd-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  auto convPath = matchedPath[kConvName];
  auto baPath = matchedPath[BIASADD_NAME];
  auto &convNode = graph->nodes.at(convPath->nodeIdx);
  auto &baNode = graph->nodes.at(baPath->nodeIdx);
  // add/biasadd node the second tensor is not constant tensor, don't fusion
  auto baNodeInputIndex = baNode->inputIndex;
  if (baNodeInputIndex.size() != BIASADD_OP_INPUT_NUM) {
    MS_LOG(ERROR) << baNode->name.c_str() << " node tensors number is invalid! ";
    return RET_ERROR;
  }
  auto baNodeBiasTensor = graph->allTensors.at(baNodeInputIndex[BIASADD_OP_CONST_TENSOR_INDEX]).get();
  MS_ASSERT(baNodeBiasTensor != nullptr);
  if (baNodeBiasTensor->nodeType != schema::NodeType_ValueNode) {
    // dont fusion, return
    return RET_OK;
  }

  // 1. generate newBiasTensor for conv
  auto status = GenConvBiasTensor(convPath, baPath, graph);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "GenConvBiasTensor failed, " << status;
    return status;
  }
  if (this->newBiasTensor != nullptr) {
    status = AddTensor2Node(graph, convPath->nodeIdx, std::move(this->newBiasTensor));
    this->newBiasTensor = nullptr;
    if (status != RET_OK) {
      MS_LOG(ERROR) << "AddTensor2Node failed,  node: " << convPath->nodeIdx << ", error: " << status;
      return status;
    }
    // add bias quantParam
    // todo add quantParam for tensors

    //    if (convNode->quantParam.size() == convNode->inputIndex.size() + convNode->outputIndex.size() - 1) {
    //      std::unique_ptr<QuantParamArrayT> quantParamArray(new QuantParamArrayT());
    //      if (quantParamArray == nullptr) {
    //        MS_LOG(ERROR) << "new QuantParamArrayT failed");
    //        return RET_ERROR;
    //      }
    //      std::unique_ptr<QuantParamT> quantParam(new QuantParamT());
    //      if (quantParam == nullptr) {
    //        MS_LOG(ERROR) << "new QuantParamT failed");
    //        return RET_ERROR;
    //      }
    //      quantParam->numBits = -1;
    //      quantParam->scale = FLT_MAX;
    //      quantParam->zeroPoint = 0;
    //      quantParam->narrowRange = true;
    //      quantParam->min = FLT_MAX;
    //      quantParam->max = FLT_MAX;
    //      quantParamArray->param.emplace_back(quantParam.release());
    //      convNode->quantParam.emplace_back(quantParamArray.release());
    //    }
  }

  // 2. change attr of conv
  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    convNode->primitive->value.AsConv2D()->hasBias = true;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    convNode->primitive->value.AsDepthwiseConv2D()->hasBias = true;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DeConv2D) {
    convNode->primitive->value.AsDeConv2D()->hasBias = true;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << convNode->primitive->value.type;
    return RET_ERROR;
  }

  // 5. delete BiasAdd node
  MergeNodeAttrFromPost(convNode, baNode);
  status = IsolateOneWayNode(graph, baPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, graph: %zu, node: %zu, error: %d";
    //, baPath->subGraphIdx, baPath->nodeIdx, status);
    return status;
  }

  return RET_OK;
}

#define BIASADD_WEIGHT_SHAPE_SIZE 1
#define BIASADD_BIAS_DIM_INDEX 0

STATUS ConvBiasAddFusionPass::GenConvBiasTensor(std::shared_ptr<Path> convPath, std::shared_ptr<Path> baPath,
                                                MetaGraphT *graph) {
  MS_ASSERT(convPath != nullptr);
  MS_ASSERT(baPath != nullptr);
  MS_ASSERT(graph != nullptr);

  auto convNode = graph->nodes.at(convPath->nodeIdx).get();
  MS_ASSERT(convNode != nullptr);
  auto baNode = graph->nodes.at(baPath->nodeIdx).get();
  MS_ASSERT(baNode != nullptr);
  int32_t kernelNum = 0;
  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    kernelNum = convNode->primitive->value.AsConv2D()->channelOut;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    kernelNum = convNode->primitive->value.AsDepthwiseConv2D()->channelIn *
                convNode->primitive->value.AsDepthwiseConv2D()->channelMultiplier;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DeConv2D) {
    kernelNum = convNode->primitive->value.AsDeConv2D()->channelOut;
  }
  auto convWeightTensorIdxes = convNode->inputIndex;
  if (convWeightTensorIdxes.size() < CONV_OP_NO_BIAS_INPUT_NUM) {
    MS_LOG(ERROR) << convNode->name.c_str() << " node tensors number is invalid! ";
    return RET_ERROR;
  }
  convWeightTensorIdxes.erase(convWeightTensorIdxes.begin());
  auto baWeightTensorIdxes = baNode->inputIndex;
  if (baWeightTensorIdxes.size() != BIASADD_OP_INPUT_NUM) {
    MS_LOG(ERROR) << baNode->name.c_str() << " node tensors number is invalid! ";
    return RET_ERROR;
  }
  baWeightTensorIdxes.erase(baWeightTensorIdxes.begin());

  if (convWeightTensorIdxes.empty()) {
    MS_LOG(ERROR) << "Conv2D should has one weight tensors at least, current number of weight tensors "
                  << convWeightTensorIdxes.size();
    return RET_ERROR;
  }

  if (baWeightTensorIdxes.empty()) {
    MS_LOG(ERROR) << "BiasAdd should has one weight tensors at least, current number of weight tensors "
                  << baWeightTensorIdxes.size();
    return RET_ERROR;
  }

  TensorT *oldBiasTensor = nullptr;
  TensorT *biasTensor = nullptr;

  if (convWeightTensorIdxes.size() == CONV_OP_HAS_BIAS_WEIGHT_NUM) {
    oldBiasTensor = graph->allTensors.at(convWeightTensorIdxes[CONV_OP_BIAS_INDEX_IN_WEIGHT]).get();
    MS_ASSERT(oldBiasTensor != nullptr);
  }
  biasTensor = graph->allTensors.at(baWeightTensorIdxes.at(BIASADD_OP_BIAS_INDEX_IN_WEIGHT)).get();
  MS_ASSERT(biasTensor != nullptr);
  auto biasDims = biasTensor->dims;
  // if biasTensor is a scaler
  if (biasDims.empty() && biasTensor->data.data() == nullptr) {
    MS_LOG(ERROR) << "BiasAdd node %s bias tensor is invalid" << baNode->name.c_str();
    return RET_ERROR;
  }
  if (!biasDims.empty() && biasDims.size() != BIASADD_WEIGHT_SHAPE_SIZE) {
    MS_LOG(ERROR) << "BiasAdd bias tensor should has one dimension, current number of dimension " << biasDims.size()
                  << ". or bias tensor is a scaler";
    return RET_ERROR;
  }

  bool bias_const = !biasDims.empty() && biasDims.size() == 1 && biasDims[0] == 1;
  if (!biasDims.empty() && !bias_const && biasDims.at(BIASADD_BIAS_DIM_INDEX) != kernelNum) {
    MS_LOG(ERROR) << "Size(%d) of BiasAdd(%s) bias tensor should be equal to kernelNum(%d)"
                  << biasDims.at(BIASADD_BIAS_DIM_INDEX) << baNode->name.c_str() << kernelNum;
    return RET_ERROR;
  }

  // cal new biasData
  this->newBiasData = new (std::nothrow) float[kernelNum];
  if (newBiasData == nullptr) {
    MS_LOG(ERROR) << "new newBiasData failed";
    return RET_ERROR;
  }

  if (biasDims.empty() && biasTensor->data.data() != nullptr) {
    auto *biasData = reinterpret_cast<float *>(biasTensor->data.data());
    if (0 != memset_s(newBiasData, kernelNum * sizeof(float), *biasData, kernelNum * sizeof(float))) {
      MS_LOG(ERROR) << "memset_s newBiasData failed";
      return RET_ERROR;
    }
  } else if (bias_const) {
    auto *biasData = reinterpret_cast<float *>(biasTensor->data.data());
    for (size_t i = 0; i < kernelNum; i++) {
      newBiasData[i] = *biasData;
    }
  } else {
    if (0 != memcpy_s(newBiasData, kernelNum * sizeof(float), biasTensor->data.data(), kernelNum * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s newBiasData failed";
      return RET_ERROR;
    }
  }
  if (oldBiasTensor != nullptr) {
    auto oldBiasDims = oldBiasTensor->dims;
    if (oldBiasDims.size() != 1) {
      MS_LOG(ERROR)
        << "Conv bias tensor should has one dimension, current number of dimension %zu";  // oldBiasDims.size());
      return RET_ERROR;
    }
    if (oldBiasDims.at(0) != kernelNum) {
      MS_LOG(ERROR)
        << "Size(%zu) of Conv bias tensor should be equal to kernelNum(%d), current number of dimension %zu";
      //              oldBiasDims.size(), kernelNum);
      return RET_ERROR;
    }
    auto *oldBiasData = reinterpret_cast<float *>(oldBiasTensor->data.data());
    for (size_t i = 0; i < kernelNum; i++) {
      oldBiasData[i] += newBiasData[i];
    }
  } else {
    auto *newCharBiasData = reinterpret_cast<uint8_t *>(newBiasData);
    std::vector<uint8_t> tmpBiasVec(newCharBiasData, newCharBiasData + kernelNum * sizeof(float) / sizeof(uint8_t));

    auto weightTensor = graph->allTensors.at(convWeightTensorIdxes[CONV_OP_FILTER_INDEX_IN_WEIGHT]).get();
    this->newBiasTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
    // todo biasShape
    this->newBiasTensor->dims = {kernelNum};
    this->newBiasTensor->dataType = weightTensor->dataType;
    this->newBiasTensor->format = weightTensor->format;
    this->newBiasTensor->refCount = weightTensor->refCount;
    this->newBiasTensor->data.swap(tmpBiasVec);
    newCharBiasData = nullptr;
  }

  delete (this->newBiasData);
  newBiasData = nullptr;

  return RET_OK;
}

ConvBiasAddFusionPass::~ConvBiasAddFusionPass() {
  if (this->newBiasData != nullptr) {
    delete (this->newBiasData);
  }
}

}  // namespace lite
}  // namespace mindspore
