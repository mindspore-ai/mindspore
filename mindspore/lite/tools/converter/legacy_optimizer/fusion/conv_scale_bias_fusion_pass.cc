/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.
 * Description: mslite
 * Author: mslite
 * Create: 2019-12-13
 */

#include <cfloat>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/conv_scale_bias_fusion_pass.h"
#include "securec/include/securec.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/op_utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {

#define CONV_SCALE_BIAS_MATCH_PATH_LEN 2

// 1. generate biasTensor according to BN weightTensor
// 2. change attr of conv
// 3. delete BN node
STATUS ConvScaleBiasFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                         std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != CONV_SCALE_BIAS_MATCH_PATH_LEN) {
    MS_LOG(ERROR) << "Conv-Scale-Bias-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  auto convPath = matchedPath[kConvName];
  MS_ASSERT(convPath != nullptr);
  auto dstPath = matchedPath[DST_NAME];
  MS_ASSERT(dstPath != nullptr);
  MS_ASSERT(subGraph != nullptr);
  auto &convNode = graph->nodes.at(convPath->nodeIdx);
  MS_ASSERT(convNode != nullptr);
  auto &dstNode = graph->nodes.at(dstPath->nodeIdx);
  MS_ASSERT(dstNode != nullptr);

  // 1. generate new weightTensor and biasTensor for conv
  auto status = GenConvWeightTensors(graph, convPath, dstPath);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "GenConvWeightTensors failed, " << status;
    return status;
  }
  if (convNode->inputIndex.size() == CONV_OP_HAS_BIAS_INPUT_NUM) {
    status = ReplaceTensorOfNode(graph, convPath->nodeIdx, convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT),
                                 std::move(this->newWeightTensor));
    this->newWeightTensor = nullptr;
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ReplaceTensorOfNode failed, subGraph: " << convPath->subGraphIdx
                    << ", node: " << convPath->nodeIdx << ", tensor "
                    << convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT) << ", error: " << status;
      return status;
    }
    status = ReplaceTensorOfNode(graph, convPath->nodeIdx, convNode->inputIndex.at(CONV_OP_BIAS_INDEX_IN_INPUT),
                                 std::move(this->newBiasTensor));
    this->newBiasTensor = nullptr;
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ReplaceTensorOfNode failed, subGraph: " << convPath->subGraphIdx
                    << ", node: " << convPath->nodeIdx << ", tensor "
                    << convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT) << ", error: " << status;
      return status;
    }
  } else if (convNode->inputIndex.size() == CONV_OP_NO_BIAS_INPUT_NUM) {
    status = ReplaceTensorOfNode(graph, convPath->nodeIdx, convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT),
                                 std::move(this->newWeightTensor));
    this->newWeightTensor = nullptr;
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ReplaceTensorOfNode failed, subGraph: " << convPath->subGraphIdx
                    << ", node: " << convPath->nodeIdx << ", tensor "
                    << convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT) << ", error: " << status;
      return status;
    }
    status = AddTensor2Node(graph, convPath->nodeIdx, std::move(this->newBiasTensor));
    this->newBiasTensor = nullptr;
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ReplaceTensorOfNode failed, subGraph: " << convPath->subGraphIdx
                    << ", node: " << convPath->nodeIdx << ", tensor "
                    << convNode->inputIndex.at(CONV_OP_FILTER_INDEX_IN_INPUT) << ", error: " << status;
      return status;
    }
    // if (convNode->name == "Conv_461") {
    // }
    // add bias quantParam
    // todo use tensor quant param
    //        if (convNode->quantParam.size() == convNode->inputIndex.size() + convNode->outputIndex.size() - 1) {
    //          std::unique_ptr<QuantParamArrayT> quantParamArray(new QuantParamArrayT());
    //          if (quantParamArray == nullptr) {
    //            MS_LOG(ERROR) << "new QuantParamArrayT failed";
    //            return RET_ERROR;
    //          }
    //          std::unique_ptr<QuantParamT> quantParam(new QuantParamT());
    //          if (quantParam == nullptr) {
    //            MS_LOG(ERROR) << "new QuantParamT failed";
    //            return RET_ERROR;
    //          }
    //          quantParam->numBits = -1;
    //          quantParam->scale = FLT_MAX;
    //          quantParam->zeroPoint = 0;
    //          quantParam->narrowRange = true;
    //          quantParam->min = FLT_MAX;
    //          quantParam->max = FLT_MAX;
    //          quantParamArray->param.emplace_back(quantParam.release());
    //          convNode->quantParam.emplace_back(quantParamArray.release());
    //        }
  } else {
    MS_LOG(ERROR) << "Conv node should has 2 or 3 weight tensors rather than " << convNode->inputIndex.size();
    return RET_ERROR;
  }

  // 2. change attr of conv
  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    convNode->primitive->value.AsConv2D()->hasBias = true;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    convNode->primitive->value.AsDepthwiseConv2D()->hasBias = true;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << convNode->primitive->value.type;
    return RET_ERROR;
  }

  // 3. delete DST node
  MergeNodeAttrFromPost(convNode, dstNode);
  status = IsolateOneWayNode(graph, dstPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed,  node: " << dstPath->nodeIdx << ", error: " << status;
    return status;
  }

  return RET_OK;
}

STATUS ConvScaleBiasFusionPass::GenConvWeightTensors(schema::MetaGraphT *graph, const std::shared_ptr<Path> &convPath,
                                                     std::shared_ptr<Path> dstPath) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(convPath != nullptr);
  MS_ASSERT(dstPath != nullptr);
  MS_ASSERT(subGraph != nullptr);
  auto &convNode = graph->nodes.at(convPath->nodeIdx);
  MS_ASSERT(convNode != nullptr);
  int32_t kernelNum = -1;
  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    kernelNum = convNode->primitive->value.AsConv2D()->channelOut;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    kernelNum = convNode->primitive->value.AsDepthwiseConv2D()->channelMultiplier *
                convNode->primitive->value.AsDepthwiseConv2D()->channelIn;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << convNode->primitive->value.type;
    return RET_ERROR;
  }
  if (kernelNum <= 0) {
    MS_LOG(ERROR) << "KernelNum should be positive, " << kernelNum;
    return RET_ERROR;
  }

  this->transScale = new (std::nothrow) float[kernelNum];
  this->transBias = new (std::nothrow) float[kernelNum];

  if (transScale == nullptr) {
    MS_LOG(ERROR) << "new transScale failed";
    return RET_ERROR;
  }

  if (transBias == nullptr) {
    MS_LOG(ERROR) << "new transBias failed";
    return RET_ERROR;
  }

  if (0 != memset_s(transScale, kernelNum * sizeof(float), 0, kernelNum * sizeof(float))) {
    MS_LOG(ERROR) << "memset transScale failed";
    return RET_ERROR;
  }

  if (0 != memset_s(transBias, kernelNum * sizeof(float), 0, kernelNum * sizeof(float))) {
    MS_LOG(ERROR) << "memset transBias failed";
    return RET_ERROR;
  }

  auto status = GetTransParam(graph, dstPath, kernelNum);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "GetTransParam failed, " << status;
    return status;
  }

  status = CalConvWeightTensors(graph, convPath, kernelNum);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "GenConvWeightTensors failed, " << status;
    return status;
  }
  return RET_OK;
}

STATUS ConvScaleBiasFusionPass::CalNewWeightTensor(TensorT *oldWeightTensor, const int32_t kernelNum,
                                                   const size_t kernelSize) {
  MS_ASSERT(oldWeightTensor != nullptr);
  auto weightData = reinterpret_cast<float *>(oldWeightTensor->data.data());
  size_t kernelDataCount = kernelNum * kernelSize;
  if (kernelDataCount == 0) {
    MS_LOG(ERROR) << "KernelDataCount should be positive, " << kernelDataCount;
    return RET_ERROR;
  }
  this->newWeightData = new (std::nothrow) float[kernelDataCount];
  if (newWeightData == nullptr) {
    MS_LOG(ERROR) << "new newWeightData failed";
    return RET_ERROR;
  }

  if (0 != memset_s(newWeightData, kernelDataCount * sizeof(float), 0, kernelDataCount * sizeof(float))) {
    MS_LOG(ERROR) << "memset newWeightData failed";
    return RET_ERROR;
  }

  for (size_t i = 0; i < kernelNum; i++) {
    for (size_t j = 0; j < kernelSize; j++) {
      newWeightData[i * kernelSize + j] = weightData[i * kernelSize + j] * transScale[i];
    }
  }
  auto newCharWeightData = reinterpret_cast<uint8_t *>(newWeightData);
  std::vector<uint8_t> tmpWeightVec(newCharWeightData,
                                    newCharWeightData + kernelDataCount * sizeof(float) / sizeof(uint8_t));

  this->newWeightTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (this->newWeightTensor == nullptr) {
    MS_LOG(ERROR) << "new newWeightTensor failed";
    return RET_ERROR;
  }
  this->newWeightTensor->dims.insert(this->newWeightTensor->dims.begin(), oldWeightTensor->dims.begin(),
                                     oldWeightTensor->dims.end());
  this->newWeightTensor->dataType = oldWeightTensor->dataType;
  this->newWeightTensor->format = oldWeightTensor->format;
  this->newWeightTensor->refCount = oldWeightTensor->refCount;
  this->newWeightTensor->data.swap(tmpWeightVec);
  delete (this->newWeightData);
  newWeightData = nullptr;

  return RET_OK;
}

STATUS ConvScaleBiasFusionPass::CalNewBiasTensor(TensorT *oldWeightTensor, TensorT *oldBiasTensor,
                                                 const int32_t kernelNum) {
  MS_ASSERT(oldWeightTensor != nullptr);
  this->newBiasData = new (std::nothrow) float[kernelNum];
  if (newBiasData == nullptr) {
    MS_LOG(ERROR) << "new newBiasData failed";
    return RET_ERROR;
  }
  if (0 != memset_s(newBiasData, kernelNum * sizeof(float), 0, kernelNum * sizeof(float))) {
    MS_LOG(ERROR) << "memset newBiasData failed";
    return RET_ERROR;
  }

  if (oldBiasTensor != nullptr) {
    auto *biasData = reinterpret_cast<float *>(oldBiasTensor->data.data());

    for (size_t i = 0; i < kernelNum; i++) {
      this->newBiasData[i] = biasData[i] * transScale[i] + transBias[i];
    }
  } else {
    if (0 != memcpy_s(newBiasData, kernelNum * sizeof(float), transBias, kernelNum * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s newBiasData failed";
      return RET_ERROR;
    }
  }
  auto *newCharBiasData = reinterpret_cast<uint8_t *>(newBiasData);
  std::vector<uint8_t> tmpBiasVec(newCharBiasData, newCharBiasData + kernelNum * sizeof(float) / sizeof(uint8_t));

  this->newBiasTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (this->newBiasTensor == nullptr) {
    MS_LOG(ERROR) << "new newBiasTensor failed";
    return RET_ERROR;
  }
  // todo biasShape
  this->newBiasTensor->dims = {kernelNum};
  this->newBiasTensor->dataType = oldWeightTensor->dataType;
  this->newBiasTensor->format = oldWeightTensor->format;
  this->newBiasTensor->refCount = oldWeightTensor->refCount;
  this->newBiasTensor->data.swap(tmpBiasVec);
  delete (this->newBiasData);
  newCharBiasData = nullptr;
  newBiasData = nullptr;
  return RET_OK;
}

STATUS ConvScaleBiasFusionPass::CalConvWeightTensors(schema::MetaGraphT *graph, const std::shared_ptr<Path> &convPath,
                                                     int32_t kernelNum) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(convPath != nullptr);

  auto convNode = graph->nodes.at(convPath->nodeIdx).get();
  MS_ASSERT(convNode != nullptr);
  auto convWeightTensorIdxes = convNode->inputIndex;
  convWeightTensorIdxes.erase(convWeightTensorIdxes.begin());

  TensorT *weightTensor = nullptr;
  TensorT *biasTensor = nullptr;
  if (convWeightTensorIdxes.size() == CONV_OP_NO_BIAS_WEIGHT_NUM) {
    weightTensor = graph->allTensors.at(convWeightTensorIdxes[CONV_OP_FILTER_INDEX_IN_WEIGHT]).get();
  } else if (convWeightTensorIdxes.size() == CONV_OP_HAS_BIAS_WEIGHT_NUM) {
    weightTensor = graph->allTensors.at(convWeightTensorIdxes[CONV_OP_FILTER_INDEX_IN_WEIGHT]).get();
    biasTensor = graph->allTensors.at(convWeightTensorIdxes[CONV_OP_BIAS_INDEX_IN_WEIGHT]).get();
  } else {
    MS_LOG(ERROR) << "Conv2D should has " << CONV_OP_NO_BIAS_WEIGHT_NUM << " or " << CONV_OP_HAS_BIAS_WEIGHT_NUM
                  << " weight tensors, current number of weight tensors " << convWeightTensorIdxes.size();
    return RET_ERROR;
  }
  if (weightTensor == nullptr) {
    MS_LOG(ERROR) << "Conv2D's weight tensor is nullptr";
    return RET_ERROR;
  }

  auto weightShape = weightTensor->dims;
  if (weightShape.size() != CONV_FILTER_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Size of dims of weight tensor should be " << CONV_FILTER_SHAPE_SIZE << " rather than "
                  << weightShape.size();
    return RET_ERROR;
  }
  size_t kernelSize = GetShapeSize(*weightTensor) / kernelNum;

  // cal new weightData
  auto status = CalNewWeightTensor(weightTensor, kernelNum, kernelSize);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CalNewWeightTensor error " << status;
    return status;
  }
  // cal new biasData
  status = CalNewBiasTensor(weightTensor, biasTensor, kernelNum);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CalNewBiasTensor error " << status;
    return status;
  }
  return RET_OK;
}

STATUS ConvScaleBiasFusionPass::Run(schema::MetaGraphT *graph) { return FusionPass::Run(graph); }

ConvScaleBiasFusionPass::~ConvScaleBiasFusionPass() {
  if (this->transScale != nullptr) {
    delete (this->transScale);
  }
  if (this->transBias != nullptr) {
    delete (this->transBias);
  }
  if (this->newWeightData != nullptr) {
    delete (this->newWeightData);
  }
  if (this->newBiasData != nullptr) {
    delete (this->newBiasData);
  }
}

}  // namespace lite
}  // namespace mindspore
