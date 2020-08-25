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

#include "tools/converter/legacy_optimizer/fusion/batchnorm_convert_scale_pass.h"
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
#define CAFFE_BATCHNORM_OP_WEIGHT_NUM 2
#define TF_BATCHNORM_OP_WEIGHT_NUM 4
#define CAFFE_BATCHNORM_MEAN_INDEX 0
#define CAFFE_BATCHNORM_VARIANCE_INDEX 1
#define TF_BATCHNORM_SCALE_INDEX 0
#define TF_BATCHNORM_BIAS_INDEX 1
#define TF_BATCHNORM_MEAN_INDEX 2
#define TF_BATCHNORM_VARIANCE_INDEX 3
namespace {
constexpr const float EPS = 1e-8;
constexpr const float EPS_DEFAULT_FLOAT = 1e-8;
constexpr const float POW_NUM = 0.5;
constexpr const int32_t NCHW_DIM_C = 1;
}
STATUS BatchNormConvertScalePass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS BatchNormConvertScalePass::DefinePattern() {
  // with preNode
  {
    auto inputOp = std::make_shared<PatternOp>();
    inputOp->id = inputOpName;
    inputOp->types = {schema::PrimitiveType_NONE};
    inputOp->isPlaceHold = true;

    auto bnOp = std::make_shared<PatternOp>();
    bnOp->id = bnOpName;
    bnOp->types = {schema::PrimitiveType_FusedBatchNorm, schema::PrimitiveType_BatchNorm};
    bnOp->left = inputOp;

    std::unique_ptr<FusionPattern> fusionPattern(new(std::nothrow) FusionPattern(bnPatternName));
    if (fusionPattern == nullptr) {
      MS_LOG(ERROR) << "new fusionPattern failed";
      return RET_ERROR;
    }
    fusionPattern->AddPatternOp(inputOp);
    fusionPattern->AddPatternOp(bnOp);
    fusionPattern->Finish();

    this->patterns.emplace_back(fusionPattern.release());
  }

  return RET_OK;
}
STATUS BatchNormConvertScalePass::DoFusion(MetaGraphT *graph, const std::string &patternName,
                                           std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (patternName != bnPatternName) {
    MS_LOG(ERROR) << "BatchNormConvertScale-Fusion match failed";
    return RET_PARAM_INVALID;
  }
  auto status = FindNodes(graph, matchedPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FindNodes failed: " << status;
    return status;
  }
  auto type = bnNode->primitive->value.type;
  if (type != schema::PrimitiveType_FusedBatchNorm && type != schema::PrimitiveType_BatchNorm) {
    return RET_OK;
  }
  auto bnPath = matchedPath.at(bnOpName);
  status = GenNewScaleTensor(graph, bnPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GenNewScaleTensor failed: " << status;
    delete[] transScale;
    delete[] transBias;
    transScale = nullptr;
    transBias = nullptr;
    return status;
  }

  status = ConvertBNToScale(graph, bnPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GenNewScaleTensor failed: " << status;
    delete[] transScale;
    delete[] transBias;
    transScale = nullptr;
    transBias = nullptr;
    return status;
  }
  delete[] transScale;
  delete[] transBias;
  transScale = nullptr;
  transBias = nullptr;
  return RET_OK;
}
STATUS BatchNormConvertScalePass::ConvertBNToScale(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath) {
  auto scaleNode = std::unique_ptr<CNodeT>(new(std::nothrow) CNodeT);
  if (scaleNode == nullptr) {
    MS_LOG(ERROR) << "new TransNode failed";
    return RET_ERROR;
  }
  scaleNode->name = bnNode->name;
  scaleNode->primitive = std::make_unique<schema::PrimitiveT>();
  if (scaleNode->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }
  scaleNode->primitive->value.type = schema::PrimitiveType_Scale;
  std::unique_ptr<ScaleT> scaleParam(new ScaleT());
  if (scaleParam == nullptr) {
    MS_LOG(ERROR) << "new transposeParam failed";
    return RET_ERROR;
  }
  scaleParam->axis = NCHW_DIM_C;
  scaleNode->primitive->value.value = scaleParam.release();
  auto scaleIter = graph->nodes.begin() + bnPath->nodeIdx;
  STATUS errorCode = RET_OK;
  scaleIter =
      InsertNode(graph, scaleIter, kBefore, 0, std::move(scaleNode), &errorCode, ScaleOpCopyer);
  if (errorCode != RET_OK) {
    MS_LOG(ERROR) << "InsertNode failed: %d";  // errorCode);
    return errorCode;
  }
  auto &newScaleNode = *(scaleIter - 1);
  graph->allTensors.emplace_back(std::move(newScaleWeightTensor));
  auto weightTensorIdx = graph->allTensors.size() - 1;
  graph->allTensors.emplace_back(std::move(newScaleBiasTensor));
  auto biasTensorIdx = graph->allTensors.size() - 1;
  newScaleNode->inputIndex.push_back(weightTensorIdx);
  newScaleNode->inputIndex.push_back(biasTensorIdx);
  // delete bn node
  auto status = IsolateOneWayNode(graph, bnPath->nodeIdx + 1, true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode " << bnNode->name.c_str() << " failed, error: " << status;
    return status;
  }
  return RET_OK;
}
STATUS BatchNormConvertScalePass::GenNewScaleTensor(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath) {
  MS_ASSERT(graph != nullptr);
  GetTransParam(graph, bnPath);
  newScaleWeightTensor = std::unique_ptr<TensorT>(new(std::nothrow) TensorT);
  if (newScaleWeightTensor == nullptr) {
    MS_LOG(ERROR) << "new weightTensor failed";
    return RET_ERROR;
  }
  newScaleWeightTensor->dataType = bnMeanTensor->dataType;
  newScaleWeightTensor->format = bnMeanTensor->format;
  newScaleWeightTensor->refCount = schema::NodeType_ValueNode;
  newScaleWeightTensor->dims = bnMeanTensor->dims;
  auto weightShapeSize = GetShapeSize(*bnMeanTensor);
  newScaleWeightTensor->data.resize(weightShapeSize * sizeof(float));
  auto ret = memcpy_s(newScaleWeightTensor->data.data(), weightShapeSize * sizeof(float), transScale,
                      weightShapeSize * sizeof(float));
  if (ret != RET_OK) {
    delete transScale;
    MS_LOG(ERROR) << "memcpy error: " << ret;
    return RET_ERROR;
  }

  newScaleBiasTensor = std::unique_ptr<TensorT>(new(std::nothrow) TensorT);
  if (newScaleBiasTensor == nullptr) {
    MS_LOG(ERROR) << "new weightTensor failed";
    return RET_ERROR;
  }
  newScaleBiasTensor->dataType = bnMeanTensor->dataType;
  newScaleBiasTensor->format = bnMeanTensor->format;

  newScaleBiasTensor->refCount = schema::NodeType_ValueNode;
  newScaleBiasTensor->dims = bnMeanTensor->dims;
  weightShapeSize = GetShapeSize(*bnMeanTensor);
  newScaleBiasTensor->data.resize(weightShapeSize * sizeof(float));
  ret = memcpy_s(newScaleBiasTensor->data.data(), weightShapeSize * sizeof(float), transBias,
                 weightShapeSize * sizeof(float));
  if (ret != RET_OK) {
    delete transBias;
    MS_LOG(ERROR) << "memcpy error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS BatchNormConvertScalePass::FindNodes(MetaGraphT *graph,
                                            const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  auto inputPath = matchedPath.at(inputOpName);
  auto bnPath = matchedPath.at(bnOpName);
  MS_ASSERT(inputPath != nullptr);
  MS_ASSERT(bnPath != nullptr);
  if (inputPath->subGraphIdx != bnPath->subGraphIdx) {
    MS_LOG(ERROR) << "matched nodes should from same subGraph";
    return RET_ERROR;
  }
  MS_ASSERT(graph->nodes.size() > inputPath->nodeIdx);
  MS_ASSERT(graph->nodes.size() > bnPath->nodeIdx);
  inputNode = graph->nodes.at(inputPath->nodeIdx).get();
  bnNode = graph->nodes.at(bnPath->nodeIdx).get();
  MS_ASSERT(inputNode != nullptr);
  MS_ASSERT(bnNode != nullptr);
  return RET_OK;
}
STATUS BatchNormConvertScalePass::GetTransParam(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnPath != nullptr);

  BNWeightTensors bnWeightTensors;

  auto status = GetBnWeightTensors(graph, bnPath, &bnWeightTensors);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetBnWeightTensors error";
    return status;
  }
  auto *meanTensor = bnWeightTensors.meanTensor;
  auto *varianceTensor = bnWeightTensors.varianceTensor;
  auto *scaleTensor = bnWeightTensors.scaleTensor;
  auto *biasTensor = bnWeightTensors.biasTensor;

  auto *meanData = reinterpret_cast<float *>(meanTensor->data.data());
  auto *varianceData = reinterpret_cast<float *>(varianceTensor->data.data());

  eps = EPS_DEFAULT_FLOAT;
  status = GetBnEpsilon(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetBnEpsilon failed";
    return status;
  }
  this->transScale = new(std::nothrow) float[bnChannel];
  this->transBias = new(std::nothrow) float[bnChannel];
  // cal transScale, tf : scale/sqrt(variance + eps); caffe : 1/sqrt(variance + eps)
  if (memcpy_s(transScale, bnChannel * sizeof(float), varianceData, bnChannel * sizeof(float)) != 0) {
    MS_LOG(ERROR) << "memcpy_s transScale error";
    delete[] transScale;
    delete[] transBias;
    transScale = nullptr;
    transBias = nullptr;
    return RET_ERROR;
  }
  // 1/sqrt(variance + eps)
  for (uint32_t i = 0; i < bnChannel; i++) {
    float tmp = transScale[i] + eps;
    tmp = pow(tmp, POW_NUM);
    transScale[i] = 1 / tmp;
  }

  if (scaleTensor != nullptr) {
    auto *scaleData = reinterpret_cast<float *>(scaleTensor->data.data());
    // scale/sqrt(variance + eps)
    for (uint32_t i = 0; i < bnChannel; i++) {
      transScale[i] *= scaleData[i];
    }
  }

  // cal transBias, tf : -scale*mean/sqrt(variance + eps) + bias; caffe : -mean/sqrt(variance + eps)
  // -mean/sqrt(variance + eps)
  for (uint32_t i = 0; i < bnChannel; i++) {
    transBias[i] = -meanData[i] * transScale[i];
  }

  if (biasTensor != nullptr) {
    auto *biasData = reinterpret_cast<float *>(biasTensor->data.data());
    // -scale*mean/sqrt(variance + eps) + bias
    for (uint32_t i = 0; i < bnChannel; i++) {
      transBias[i] += biasData[i];
    }
  }

  return RET_OK;
}

// BatchNorm weight Tensor definition:
// caffe
//   estimated_mean  --0
//   estimated_variance  --1
// tensorflow
//   scale    -- 0
//   bias        --1
//   estimated_mean  --2
//   estimated_variance  --3
STATUS BatchNormConvertScalePass::GetBnWeightTensors(MetaGraphT *graph, const std::shared_ptr<Path> &bnPath,
                                                     BNWeightTensors* bnWeightTensors) {
  if (graph == nullptr || bnPath == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_NULL_PTR;
  }
  MS_ASSERT(graph->allTensors.size() > bnNode->inputIndex.at(1));
  auto bnWeightTensorIdxes = bnNode->inputIndex;
  bnWeightTensorIdxes.erase(bnWeightTensorIdxes.begin());
  if (bnWeightTensorIdxes.size() == CAFFE_BATCHNORM_OP_WEIGHT_NUM) {
    bnWeightTensors->meanTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors->varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_VARIANCE_INDEX]).get();
  } else if (bnWeightTensorIdxes.size() == TF_BATCHNORM_OP_WEIGHT_NUM) {
    bnWeightTensors->scaleTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_SCALE_INDEX]).get();
    bnWeightTensors->biasTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_BIAS_INDEX]).get();
    bnWeightTensors->meanTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors->varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_VARIANCE_INDEX]).get();
  } else {
    MS_LOG(ERROR) << "BatchNorm should has 2 or 4 weight tensors, current number of weight tensors: "
                  << bnWeightTensorIdxes.size();
    return RET_ERROR;
  }

  if (bnWeightTensors->meanTensor == nullptr) {
    MS_LOG(ERROR) << "BatchNorm's mean tensor is nullptr";
    return RET_ERROR;
  }

  if (bnWeightTensors->varianceTensor == nullptr) {
    MS_LOG(ERROR) << "BatchNorm's variance tensor is nullptr";
    return RET_ERROR;
  }
  bnChannel = bnWeightTensors->meanTensor->data.size() * sizeof(uint8_t) / sizeof(float);
  if (bnChannel <= 0) {
    MS_LOG(ERROR) << "BatchNorm's channel less or equal 0";
    return RET_ERROR;
  }
  bnMeanTensor = bnWeightTensors->meanTensor;
  if (bnChannel != bnWeightTensors->varianceTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
    MS_LOG(ERROR) << "conv kernel num expected to be equal to variance size";
    return RET_ERROR;
  }

  if (bnWeightTensors->scaleTensor != nullptr) {
    if (bnChannel != bnWeightTensors->scaleTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
      MS_LOG(ERROR) << "conv kernel num  expected to be equal to scale size";
      return RET_ERROR;
    }
  }

  if (bnWeightTensors->biasTensor != nullptr) {
    if (bnChannel != bnWeightTensors->biasTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
      MS_LOG(ERROR) << "conv kernel num expected to be equal to bias size";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS BatchNormConvertScalePass::GetBnEpsilon(MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_NULL_PTR;
  }
  if (bnNode == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_NULL_PTR;
  }
  if (bnNode->primitive->value.type == schema::PrimitiveType_FusedBatchNorm) {
    eps = bnNode->primitive->value.AsFusedBatchNorm()->epsilon;
  } else if (bnNode->primitive->value.type == schema::PrimitiveType_BatchNorm) {
    eps = bnNode->primitive->value.AsBatchNorm()->epsilon;
  } else {
    MS_LOG(ERROR) << "match pattern has error, not BatchNorm node";
    return RET_ERROR;
  }

  if (eps < EPS) {
    eps = EPS_DEFAULT_FLOAT;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
