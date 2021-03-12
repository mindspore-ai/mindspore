/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/batchnorm_convert_scale_pass.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tools/converter/converter_flags.h"
#include "third_party/securec/include/securec.h"
#include "src/common/log_adapter.h"
#include "tools/common/tensor_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define CAFFE_BATCHNORM_MEAN_INDEX 0
#define CAFFE_BATCHNORM_VARIANCE_INDEX 1
#define CAFFE_BATCHNORM_SCALE_INDEX 2
#define TF_BATCHNORM_SCALE_INDEX 0
#define TF_BATCHNORM_BIAS_INDEX 1
#define TF_BATCHNORM_MEAN_INDEX 2
#define TF_BATCHNORM_VARIANCE_INDEX 3
namespace {
constexpr const float EPS = 1e-8;
constexpr const float EPS_DEFAULT_FLOAT = 1e-8;
constexpr const float POW_NUM = 0.5;
constexpr uint32_t kQuadrupleNum = 4;
}  // namespace

STATUS BatchNormConvertScalePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);

  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    auto type = node->primitive->value.type;
    if (type != schema::PrimitiveType_FusedBatchNorm && type != schema::PrimitiveType_BatchNorm) {
      continue;
    }

    auto input_index = node->inputIndex.at(0);
    if (graph->allTensors.at(input_index)->dims.empty()) {
      MS_LOG(WARNING) << "The shape of input tensor is uncertain.";
      return RET_OK;
    }
    auto status = GenNewScaleTensor(graph, node);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "GenNewScaleTensor failed: " << status;
      return status;
    }
    status = ConvertBNToScale(graph, node);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "GenNewScaleTensor failed: " << status;
      return status;
    }
  }
  return RET_OK;
}
STATUS BatchNormConvertScalePass::ConvertBNToScale(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnNode != nullptr);
  bnNode->primitive->value.type = schema::PrimitiveType_ScaleFusion;
  std::unique_ptr<ScaleFusionT> scaleParam(new (std::nothrow) ScaleFusionT());
  if (scaleParam == nullptr) {
    MS_LOG(ERROR) << "new scaleParam failed";
    return RET_ERROR;
  }
  //  after fusion bn must NHWC
  auto input0 = bnNode->inputIndex.at(0);
  if (graph->allTensors.at(input0)->dims.size() == kQuadrupleNum) {
    scaleParam->axis = -1;
  } else {
    scaleParam->axis = 1;
  }
  bnNode->primitive->value.value = scaleParam.release();
  bnNode->inputIndex.clear();
  bnNode->inputIndex.push_back(input0);
  graph->allTensors.emplace_back(std::move(newScaleWeightTensor));
  auto weightTensorIdx = graph->allTensors.size() - 1;
  graph->allTensors.emplace_back(std::move(newScaleBiasTensor));
  auto biasTensorIdx = graph->allTensors.size() - 1;
  bnNode->inputIndex.push_back(weightTensorIdx);
  bnNode->inputIndex.push_back(biasTensorIdx);
  return RET_OK;
}
STATUS BatchNormConvertScalePass::GenNewScaleTensor(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnNode != nullptr);
  GetTransParam(graph, bnNode);
  newScaleWeightTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newScaleWeightTensor == nullptr) {
    MS_LOG(ERROR) << "new weightTensor failed";
    return RET_ERROR;
  }
  newScaleWeightTensor->dataType = bnMeanTensor->dataType;
  newScaleWeightTensor->format = bnMeanTensor->format;
  newScaleWeightTensor->refCount = NodeType_ValueNode;
  newScaleWeightTensor->dims = bnMeanTensor->dims;
  auto weightShapeSize = GetShapeSize(*bnMeanTensor);
  newScaleWeightTensor->data.resize(weightShapeSize * sizeof(float));
  auto ret = memcpy_s(newScaleWeightTensor->data.data(), weightShapeSize * sizeof(float), transScale,
                      weightShapeSize * sizeof(float));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error: " << ret;
    delete[] transScale;
    delete[] transBias;
    transScale = nullptr;
    transBias = nullptr;
    return RET_ERROR;
  }

  newScaleBiasTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newScaleBiasTensor == nullptr) {
    MS_LOG(ERROR) << "new weightTensor failed";
    return RET_ERROR;
  }
  newScaleBiasTensor->dataType = bnMeanTensor->dataType;
  newScaleBiasTensor->format = bnMeanTensor->format;

  newScaleBiasTensor->refCount = NodeType_ValueNode;
  newScaleBiasTensor->dims = bnMeanTensor->dims;
  weightShapeSize = GetShapeSize(*bnMeanTensor);
  newScaleBiasTensor->data.resize(weightShapeSize * sizeof(float));
  ret = memcpy_s(newScaleBiasTensor->data.data(), weightShapeSize * sizeof(float), transBias,
                 weightShapeSize * sizeof(float));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error: " << ret;
    delete[] transScale;
    delete[] transBias;
    transScale = nullptr;
    transBias = nullptr;
    return RET_ERROR;
  }
  delete[] transScale;
  delete[] transBias;
  transScale = nullptr;
  transBias = nullptr;
  return RET_OK;
}
STATUS BatchNormConvertScalePass::GetTransParam(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnNode != nullptr);
  BNWeightTensors bnWeightTensors;

  auto status = GetBnWeightTensors(graph, &bnWeightTensors, bnNode);
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
  status = GetBnEpsilon(bnNode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetBnEpsilon failed";
    return status;
  }
  this->transScale = new (std::nothrow) float[bnChannel];
  if (this->transScale == nullptr) {
    MS_LOG(ERROR) << "new transScale failed";
    return RET_ERROR;
  }
  this->transBias = new (std::nothrow) float[bnChannel];
  if (this->transBias == nullptr) {
    MS_LOG(ERROR) << "new transBias failed";
    return RET_ERROR;
  }
  // cal transScale, tf : scale/sqrt(variance + eps); caffe : 1/sqrt(variance + eps)
  if (memcpy_s(transScale, bnChannel * sizeof(float), varianceData, bnChannel * sizeof(float)) != EOK) {
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
    if (tmp <= 0.0f) {
      MS_LOG(ERROR) << "divisor 'tmp' cannot be 0";
      return RET_ERROR;
    }
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
STATUS BatchNormConvertScalePass::GetBnWeightTensors(MetaGraphT *graph, BNWeightTensors *bnWeightTensors,
                                                     const std::unique_ptr<CNodeT> &bnNode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnNode != nullptr);
  MS_ASSERT(bnWeightTensors != nullptr);
  MS_ASSERT(graph->allTensors.size() > bnNode->inputIndex.at(1));
  auto bnWeightTensorIdxes = bnNode->inputIndex;
  bnWeightTensorIdxes.erase(bnWeightTensorIdxes.begin());
  if (fmkType == converter::FmkType_CAFFE) {
    bnWeightTensors->meanTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors->varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_VARIANCE_INDEX]).get();
    auto scaleTensor = graph->allTensors.at(bnWeightTensorIdxes[CAFFE_BATCHNORM_SCALE_INDEX]).get();

    // calibrate mean and variance
    float scale_factor_data = (reinterpret_cast<float *>(scaleTensor->data.data()))[0];
    float scale_factor = scale_factor_data == 0 ? 0 : 1 / scale_factor_data;
    auto mean_data = reinterpret_cast<float *>(bnWeightTensors->meanTensor->data.data());
    auto variance_data = reinterpret_cast<float *>(bnWeightTensors->varianceTensor->data.data());
    for (size_t i = 0; i < GetShapeSize(*bnWeightTensors->meanTensor); i++) {
      mean_data[i] *= scale_factor;
    }
    for (size_t i = 0; i < GetShapeSize(*bnWeightTensors->varianceTensor); i++) {
      variance_data[i] *= scale_factor;
    }
  } else {
    bnWeightTensors->scaleTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_SCALE_INDEX]).get();
    bnWeightTensors->biasTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_BIAS_INDEX]).get();
    bnWeightTensors->meanTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_MEAN_INDEX]).get();
    bnWeightTensors->varianceTensor = graph->allTensors.at(bnWeightTensorIdxes[TF_BATCHNORM_VARIANCE_INDEX]).get();
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

STATUS BatchNormConvertScalePass::GetBnEpsilon(const std::unique_ptr<CNodeT> &bnNode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(bnNode != nullptr);
  MS_ASSERT(bnNode->primitive != nullptr);
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
