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

#include "tools/converter/quantizer/aware_quantizer.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/common/utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/calc_quant_param.h"
#include "src/common/log_adapter.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
AwareQuantizer::AwareQuantizer(schema::MetaGraphT *graph, const TypeId &inferType) : FbQuantizer(graph) {}

STATUS AwareQuantizer::RemoveFakeQuant() { return RET_OK; }

STATUS AwareQuantizer::GenerateQuantParam() {
  auto *quantParamRegister = QuantParamCalcRegister::GetInstance();

  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    MS_ASSERT(node != nullptr);
    if (GetCNodeTType(*node) == schema::PrimitiveType_FakeQuantWithMinMax ||
        GetCNodeTType(*node) == schema::PrimitiveType_FakeQuantWithMinMaxVars) {
      MS_ASSERT(false);
    }
    auto quantParamCalcer = quantParamRegister->GetQuantParamCalcer(GetCNodeTType(*node));
    if (quantParamCalcer == nullptr) {
      MS_LOG(WARNING) << "Can not find QuantParamCalcer for " << node->name.c_str()
                      << ", type: " << GetCNodeTTypeName(*node).c_str() << " set node to QuantNone and skip";
      node->quantType = static_cast<schema::QuantType>(QuantType_QUANT_NONE);
    } else {
      auto status = quantParamCalcer->Calc(graph, *node);
      if (status != RET_OK) {
        MS_LOG(WARNING) << "quantParamCalcer failed: " << status << " node: " << node->name.c_str();
        node->quantType = schema::QuantType_QUANT_NONE;
      } else {
        node->quantType = schema::QuantType_AwareTraining;
      }
    }
  }
  return RET_OK;
}

STATUS AwareQuantizer::DoQuantize() {
  for (auto &tensor : graph->allTensors) {
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited || tensor->data.empty()) {
      continue;
    }
    if (tensor->dataType != TypeId::kNumberTypeFloat32 && tensor->dataType != TypeId::kNumberTypeFloat &&
        tensor->dataType != TypeId::kNumberTypeUInt8) {
      continue;
    }
    // perlayer
    if (tensor->quantParams.size() == 1) {
      auto &quantParam = tensor->quantParams.front();
      size_t wShapeSize = GetShapeSize(*(tensor.get()));
      void *oriWeightData = tensor->data.data();
      if (quantParam->dstDtype == TypeId::kNumberTypeInt8) {
        vector<int8_t> qDatas(wShapeSize);
        auto weightQauntParam = GetTensorQuantParam(tensor);
        if (tensor->dataType == TypeId::kNumberTypeFloat ||
            tensor->dataType == TypeId::kNumberTypeFloat32) {  // normal awareing quant
          auto *weightData = static_cast<float *>(oriWeightData);
          for (size_t j = 0; j < wShapeSize; j++) {
            qDatas[j] = QuantizeData<int8_t>(weightData[j], weightQauntParam.get());
          }
        } else {  // tflite awareing quant
          auto *weightData = static_cast<uint8_t *>(oriWeightData);
          for (size_t j = 0; j < wShapeSize; j++) {
            qDatas[j] = (int32_t)weightData[j] - 128;
          }
          weightQauntParam->zeroPoint -= 128;
          tensor->quantParams.clear();
          tensor->quantParams.emplace_back(weightQauntParam.release());
        }

        ::memcpy(tensor->data.data(), qDatas.data(), wShapeSize);
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
    } else {  // pertensor
    }
  }
  return RET_OK;
}
STATUS AwareQuantizer::DetermineNodeQuantType() {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    MS_ASSERT(node != nullptr);
    bool canQuant = true;
    for (auto &outTensorIdx : node->outputIndex) {
      MS_ASSERT(graph->allTensors.size() > outTensorIdx);
      auto &outTensor = graph->allTensors.at(outTensorIdx);
      MS_ASSERT(outTensor != nullptr);
      if (outTensor->quantParams.empty() || outTensor->quantParams.front() == nullptr ||
          !outTensor->quantParams.front()->inited) {
        canQuant = false;
        break;
      }
    }

    if (canQuant && IsContain(GetInt8OpList(), GetCNodeTType(*node))) {
      node->quantType = schema::QuantType_AwareTraining;
    } else {
      node->quantType = schema::QuantType_QUANT_NONE;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
