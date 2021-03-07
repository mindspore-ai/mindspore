/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <vector>
#include <cmath>
#include "tools/converter/legacy_optimizer/graph/tensor_quant_pass.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore::lite {
namespace {
STATUS PreHandleQuantDtypeCast(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    if (node == nullptr || node->primitive == nullptr) {
      MS_LOG(ERROR) << " node or node->primitive is nullptr";
      return RET_ERROR;
    }
    if (node->primitive->value.type == PrimitiveType_QuantDTypeCast) {
      auto attr = node->primitive->value.AsQuantDTypeCast();
      auto &inputTensor = graph->allTensors.at(node->inputIndex.front());
      inputTensor->dataType = attr->src_t;
      auto &outputTensor = graph->allTensors.at(node->outputIndex.front());
      outputTensor->dataType = attr->dst_t;

      if (attr->src_t == TypeId::kNumberTypeUInt8) {
        attr->src_t = TypeId::kNumberTypeInt8;
      }
      if (attr->dst_t == TypeId::kNumberTypeUInt8) {
        attr->dst_t = TypeId::kNumberTypeInt8;
      }
    }
  }
  return RET_OK;
}

STATUS ComputeDataToInt8(const std::unique_ptr<TensorT> &tensor, int32_t index) {
  MS_ASSERT(tensor != nullptr);
  size_t wShapeSize = tensor->data.empty() ? 0 : GetShapeSize(*(tensor.get()));
  void *oriWeightData = tensor->data.data();
  std::vector<int8_t> qDatas(wShapeSize);
  auto weightQauntParam = GetTensorQuantParam(tensor);
  if (tensor->dataType == TypeId::kNumberTypeFloat ||
      tensor->dataType == TypeId::kNumberTypeFloat32) {  // normal awareing quant
    auto *weightData = static_cast<float *>(oriWeightData);
    if (weightData == nullptr) {
      return RET_OK;
    }
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = quant::QuantizeData<int8_t>(weightData[j], weightQauntParam.get());
    }
  } else {  // convert uint8 to int8
    auto *weightData = static_cast<uint8_t *>(oriWeightData);
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = (int32_t)weightData[j] - 128;
    }
    weightQauntParam->zeroPoint -= 128;
    tensor->quantParams.clear();
    tensor->quantParams.emplace_back(weightQauntParam.release());
    TensorDataType::GetInstance()->UpdateTensorType(index, TypeId::kNumberTypeUInt8);
  }
  tensor->dataType = TypeId::kNumberTypeInt8;
  if (tensor->data.empty()) {
    return RET_OK;
  }
  tensor->data.clear();
  tensor->data.resize(wShapeSize * sizeof(int8_t));
  if (memcpy_s(tensor->data.data(), wShapeSize * sizeof(int8_t), qDatas.data(), wShapeSize * sizeof(int8_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ComputeDataToInt32(const std::unique_ptr<TensorT> &tensor) {
  MS_ASSERT(tensor != nullptr);
  auto bShapeSize = GetShapeSize(*(tensor));
  std::unique_ptr<int32_t[]> qDatas(new (std::nothrow) int32_t[bShapeSize]);
  if (qDatas == nullptr) {
    MS_LOG(ERROR) << "new qDatas failed";
    return RET_ERROR;
  }
  void *biasData = tensor->data.data();
  auto *rawDatas = static_cast<float *>(biasData);
  if (fabs(tensor->quantParams.front()->scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  for (size_t i = 0; i < bShapeSize; ++i) {
    qDatas[i] = (int32_t)std::round(rawDatas[i] / tensor->quantParams.front()->scale);
  }
  tensor->dataType = TypeId::kNumberTypeInt32;
  tensor->data.clear();
  tensor->data.resize(bShapeSize * sizeof(int32_t));
  if (memcpy_s(tensor->data.data(), bShapeSize * sizeof(int32_t), qDatas.get(), bShapeSize * sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

STATUS TensorQuantPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto status = PreHandleQuantDtypeCast(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "pre adjust failed.";
    return status;
  }
  int32_t index = 0;
  for (auto &tensor : graph->allTensors) {
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      index++;
      continue;
    }
    if (tensor->dataType != TypeId::kNumberTypeFloat32 && tensor->dataType != TypeId::kNumberTypeFloat &&
        tensor->dataType != TypeId::kNumberTypeUInt8 && tensor->dataType != TypeId::kTypeUnknown) {
      index++;
      continue;
    }
    if (tensor->quantParams.size() != 1) {  // perchannel
      MS_LOG(ERROR) << "perchannel doquant is not supported yet";
      return RET_ERROR;
    }
    // perlayer
    auto &quantParam = tensor->quantParams.front();
    if (quantParam->dstDtype == TypeId::kNumberTypeUInt8 || quantParam->dstDtype == TypeId::kNumberTypeFloat32 ||
        quantParam->dstDtype == TypeId::kNumberTypeFloat) {
      status = ComputeDataToInt8(tensor, index);
    } else if (quantParam->dstDtype == TypeId::kNumberTypeInt32) {
      // quant bias data
      status = ComputeDataToInt32(tensor);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "compute data to int8 or int32 failed.";
      return status;
    }
    index++;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
