/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/quant_helper/conv_quant_param_propogator.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
static constexpr size_t kBiasAdd = 3;

STATUS ConvQuantParamPropogator::PropogateQuantParams(mindspore::schema::MetaGraphT *graph,
                                                      const mindspore::schema::CNodeT &node) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  if (node.inputIndex.size() == kBiasAdd) {
    MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(kBiasAdd - 1));
    auto &bias_tensor = graph->allTensors.at(node.inputIndex.at(kBiasAdd - 1));
    if (bias_tensor->quantParams.empty() || !bias_tensor->quantParams.front()->inited) {
      // check input and weight quant params
      auto &input_tensor = graph->allTensors.at(node.inputIndex.at(0));
      auto &weight_tensor = graph->allTensors.at(node.inputIndex.at(1));
      if (input_tensor->quantParams.empty() || !input_tensor->quantParams.front()->inited) {
        return RET_OK;
      }

      if (weight_tensor->quantParams.empty() || !weight_tensor->quantParams.front()->inited) {
        return RET_OK;
      }
      auto &input_quant_param = input_tensor->quantParams.at(0);
      std::vector<std::unique_ptr<schema::QuantParamT>> bias_quant_params;
      for (auto &weight_quant_param : weight_tensor->quantParams) {
        auto bias_quant_param = std::make_unique<schema::QuantParamT>();
        MS_CHECK_TRUE_MSG(bias_quant_param != nullptr, RET_NULL_PTR, "bias_quant_param is nullptr.");
        bias_quant_param->min = 0.0;
        bias_quant_param->max = 0.0;
        bias_quant_param->dstDtype = kNumberTypeInt32;
        bias_quant_param->inited = input_quant_param->inited && weight_quant_param->inited;
        bias_quant_param->zeroPoint = 0;
        if (bias_quant_param->inited) {
          bias_quant_param->scale = input_quant_param->scale * weight_quant_param->scale;
        }
        bias_quant_param->roundType = 1;
        bias_quant_param->multiplier = 1;
        bias_quant_params.emplace_back(std::move(bias_quant_param));
      }
      bias_tensor->quantParams = std::move(bias_quant_params);
    }
    for (auto &quantParam : bias_tensor->quantParams) {
      quantParam->dstDtype = TypeId::kNumberTypeInt32;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
