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

#include <string>
#include <unordered_map>
#include <memory>
#include <cmath>
#include "tools/converter/legacy_optimizer/fusion/conv_bn_fusion_pass.h"
#include "securec/include/securec.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "utils/log_adapter.h"

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

constexpr const float EPS = 1e-8;
constexpr const float EPS_DEFAULT_FLOAT = 1e-5;
constexpr const float POW_NUM = 0.5;

STATUS ConvBNFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  return ConvScaleBiasFusionPass::DoFusion(graph, patternName, matchedPath);
}

STATUS ConvBNFusionPass::DefinePattern() {
  auto convOp = std::make_shared<PatternOp>();
  convOp->id = kConvName;
  convOp->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
  auto bnOp = std::make_shared<PatternOp>();
  bnOp->id = DST_NAME;
  bnOp->types = {schema::PrimitiveType_FusedBatchNorm, schema::PrimitiveType_BatchNorm};
  bnOp->left = convOp;

  std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern("ConvBatchNormFusion"));
  if (fusionPattern == nullptr) {
    MS_LOG(ERROR) << "new fusionPattern failed";
    return RET_ERROR;
  }
  fusionPattern->AddPatternOp(convOp);
  fusionPattern->AddPatternOp(bnOp);
  fusionPattern->Finish();

  this->patterns.emplace_back(fusionPattern.release());

  return RET_OK;
}

STATUS ConvBNFusionPass::Run(schema::MetaGraphT *graph) { return ConvScaleBiasFusionPass::Run(graph); }

STATUS ConvBNFusionPass::GetTransParam(schema::MetaGraphT *graph, std::shared_ptr<Path> bnPath, int32_t kernelNum) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnPath != nullptr);

  BNWeightTensors bnWeightTensors;

  auto status = GetBnWeightTensors(graph, bnPath, kernelNum, bnWeightTensors);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetBnWeightTensors error " << status;
    return status;
  }
  schema::TensorT *meanTensor = bnWeightTensors.meanTensor;
  schema::TensorT *varianceTensor = bnWeightTensors.varianceTensor;
  schema::TensorT *scaleTensor = bnWeightTensors.scaleTensor;
  schema::TensorT *biasTensor = bnWeightTensors.biasTensor;

  auto *meanData = reinterpret_cast<float *>(meanTensor->data.data());
  auto *varianceData = reinterpret_cast<float *>(varianceTensor->data.data());

  float eps = EPS_DEFAULT_FLOAT;
  status = GetBnEpsilon(graph, bnPath, eps);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetBnEpsilon failed " << status;
    return status;
  }

  // cal transScale, tf : scale/sqrt(variance + eps); caffe : 1/sqrt(variance + eps)
  if (memcpy_s(transScale, kernelNum * sizeof(float), varianceData, kernelNum * sizeof(float)) != 0) {
    MS_LOG(ERROR) << "memcpy_s transScale error";
    return RET_ERROR;
  }
  // 1/sqrt(variance + eps)
  for (int32_t i = 0; i < kernelNum; i++) {
    float tmp = transScale[i] + eps;
    tmp = pow(tmp, POW_NUM);
    transScale[i] = 1 / tmp;
  }

  if (scaleTensor != nullptr) {
    auto *scaleData = reinterpret_cast<float *>(scaleTensor->data.data());
    // scale/sqrt(variance + eps)
    for (int32_t i = 0; i < kernelNum; i++) {
      transScale[i] *= scaleData[i];
    }
  }

  // cal transBias, tf : -scale*mean/sqrt(variance + eps) + bias; caffe : -mean/sqrt(variance + eps)
  // -mean/sqrt(variance + eps)
  for (int32_t i = 0; i < kernelNum; i++) {
    transBias[i] = -meanData[i] * transScale[i];
  }

  if (biasTensor != nullptr) {
    auto *biasData = reinterpret_cast<float *>(biasTensor->data.data());
    // -scale*mean/sqrt(variance + eps) + bias
    for (int32_t i = 0; i < kernelNum; i++) {
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
STATUS ConvBNFusionPass::GetBnWeightTensors(schema::MetaGraphT *graph, std::shared_ptr<Path> bnPath, int32_t kernelNum,
                                            BNWeightTensors &bnWeightTensors) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnPath != nullptr);
  auto bnNode = graph->nodes.at(bnPath->nodeIdx).get();
  auto bnWeightTensorIdxes = bnNode->inputIndex;
  bnWeightTensorIdxes.erase(bnWeightTensorIdxes.begin());
  if (bnWeightTensorIdxes.size() == CAFFE_BATCHNORM_OP_WEIGHT_NUM) {
    bnWeightTensors.meanTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors.varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_VARIANCE_INDEX]).get();
  } else if (bnWeightTensorIdxes.size() == TF_BATCHNORM_OP_WEIGHT_NUM) {
    bnWeightTensors.scaleTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_SCALE_INDEX]).get();
    bnWeightTensors.biasTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_BIAS_INDEX]).get();
    bnWeightTensors.meanTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors.varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_VARIANCE_INDEX]).get();
  } else {
    MS_LOG(ERROR) << "BatchNorm should has " << CAFFE_BATCHNORM_OP_WEIGHT_NUM << " or " << TF_BATCHNORM_OP_WEIGHT_NUM
                  << " weight tensors, current number of weight tensors " << bnWeightTensorIdxes.size();
    return RET_ERROR;
  }

  if (bnWeightTensors.meanTensor == nullptr) {
    MS_LOG(ERROR) << "BatchNorm's mean tensor is nullptr";
    return RET_ERROR;
  }

  if (bnWeightTensors.varianceTensor == nullptr) {
    MS_LOG(ERROR) << "BatchNorm's variance tensor is nullptr";
    return RET_ERROR;
  }

  if (kernelNum != bnWeightTensors.meanTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
    MS_LOG(ERROR) << "conv kernel num " << kernelNum << " is expected to be equal to mean size("
                  << bnWeightTensors.meanTensor->data.size() * sizeof(uint8_t) / sizeof(float) << ")";
    return RET_ERROR;
  }

  if (kernelNum != bnWeightTensors.varianceTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
    MS_LOG(ERROR) << "conv kernel num " << kernelNum << " is expected to be equal to mean size("
                  << bnWeightTensors.meanTensor->data.size() * sizeof(uint8_t) / sizeof(float) << ")";
    return RET_ERROR;
  }

  if (bnWeightTensors.scaleTensor != nullptr) {
    if (kernelNum != bnWeightTensors.scaleTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
      MS_LOG(ERROR) << "conv kernel num " << kernelNum << " is expected to be equal to mean size("
                    << bnWeightTensors.meanTensor->data.size() * sizeof(uint8_t) / sizeof(float) << ")";
      return RET_ERROR;
    }
  }

  if (bnWeightTensors.biasTensor != nullptr) {
    if (kernelNum != bnWeightTensors.biasTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
      MS_LOG(ERROR) << "conv kernel num " << kernelNum << " is expected to be equal to mean size("
                    << bnWeightTensors.meanTensor->data.size() * sizeof(uint8_t) / sizeof(float) << ")";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS ConvBNFusionPass::GetBnEpsilon(schema::MetaGraphT *graph, std::shared_ptr<Path> bnPath, float &eps) {
  MS_ASSERT(graph != nullptr);
  auto bnNode = graph->nodes.at(bnPath->nodeIdx).get();
  MS_ASSERT(bnNode != nullptr);
  if (bnNode->primitive->value.type == schema::PrimitiveType_FusedBatchNorm) {
    eps = bnNode->primitive->value.AsFusedBatchNorm()->epsilon;
  } else if (bnNode->primitive->value.type == schema::PrimitiveType_BatchNorm) {
    eps = bnNode->primitive->value.AsBatchNorm()->epsilon;
  } else {
    MS_LOG(ERROR) << "match pattern has error, " << bnNode->name.c_str() << " not BatchNorm node";
    return RET_ERROR;
  }

  if (eps < EPS) {
    eps = EPS_DEFAULT_FLOAT;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
