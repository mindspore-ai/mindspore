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
STATUS TensorQuantPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    if (node == nullptr || node->primitive == nullptr) {
      MS_LOG(ERROR) << " node or node->primitive is nullptr";
      return RET_ERROR;
    }
    if (node->primitive->value.type == PrimitiveType_QuantDTypeCast) {
      auto attr = node->primitive->value.AsQuantDTypeCast();
      auto &inputTensor = graph->allTensors.at(node->inputIndex.front());
      inputTensor->dataType = attr->srcT;
      auto &outputTensor = graph->allTensors.at(node->outputIndex.front());
      outputTensor->dataType = attr->dstT;

      if (attr->srcT == TypeId::kNumberTypeUInt8) {
        attr->srcT = TypeId::kNumberTypeInt8;
      }
      if (attr->dstT == TypeId::kNumberTypeUInt8) {
        attr->dstT = TypeId::kNumberTypeInt8;
      }
    }
  }
  unsigned int index = -1;
  for (auto &tensor : graph->allTensors) {
    index++;
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      continue;
    }
    if (tensor->dataType != TypeId::kNumberTypeFloat32 && tensor->dataType != TypeId::kNumberTypeFloat &&
        tensor->dataType != TypeId::kNumberTypeUInt8 && tensor->dataType != TypeId::kTypeUnknown) {
      continue;
    }
    // perlayer
    if (tensor->quantParams.size() == 1) {
      auto &quantParam = tensor->quantParams.front();
      size_t wShapeSize = tensor->data.empty() ? 0 : GetShapeSize(*(tensor.get()));
      void *oriWeightData = tensor->data.data();
      if (quantParam->dstDtype == TypeId::kNumberTypeUInt8 || quantParam->dstDtype == TypeId::kNumberTypeFloat32 ||
          quantParam->dstDtype == TypeId::kNumberTypeFloat) {
        std::vector<int8_t> qDatas(wShapeSize);
        auto weightQauntParam = GetTensorQuantParam(tensor);
        if (tensor->dataType == TypeId::kNumberTypeFloat ||
            tensor->dataType == TypeId::kNumberTypeFloat32) {  // normal awareing quant
          auto *weightData = static_cast<float *>(oriWeightData);
          if (weightData == nullptr) {
            continue;
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
        if (!tensor->data.empty()) {
          tensor->data.clear();
          tensor->data.resize(wShapeSize * sizeof(int8_t));
          auto ret =
            memcpy_s(tensor->data.data(), wShapeSize * sizeof(int8_t), qDatas.data(), wShapeSize * sizeof(int8_t));
          if (ret != EOK) {
            MS_LOG(ERROR) << "memcpy_s failed: " << ret;
            return RET_ERROR;
          }
        }
      } else if (quantParam->dstDtype == TypeId::kNumberTypeInt32) {
        // quant bias data
        auto bShapeSize = GetShapeSize(*(tensor.get()));
        std::unique_ptr<int32_t[]> qDatas(new (std::nothrow) int32_t[bShapeSize]);
        if (qDatas == nullptr) {
          MS_LOG(ERROR) << "new qDatas failed";
          return RET_ERROR;
        }
        void *biasData = tensor->data.data();
        auto *rawDatas = static_cast<float *>(biasData);
        if (fabs(quantParam->scale) <= 0.0f) {
          MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
          return RET_ERROR;
        }
        for (size_t i = 0; i < bShapeSize; ++i) {
          qDatas[i] = (int32_t)std::round(rawDatas[i] / quantParam->scale);
        }
        tensor->dataType = TypeId::kNumberTypeInt32;
        tensor->data.clear();
        tensor->data.resize(bShapeSize * sizeof(int32_t));
        auto ret =
          memcpy_s(tensor->data.data(), bShapeSize * sizeof(int32_t), qDatas.get(), bShapeSize * sizeof(int32_t));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed: " << ret;
          return RET_ERROR;
        }
      }
    } else {  // perchannel
      MS_LOG(ERROR) << "perchannel doquant is not supported yet";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
