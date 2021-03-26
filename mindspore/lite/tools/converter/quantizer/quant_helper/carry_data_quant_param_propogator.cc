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

#include "tools/converter/quantizer/quant_helper/carry_data_quant_param_propogator.h"
#include <utility>
#include <memory>
#include "tools/common/tensor_util.h"
namespace mindspore::lite {
STATUS CarryDataQuantParamPropogator::PropogateQuantParams(schema::MetaGraphT *graph, const schema::CNodeT &node) {
  UpdateQuantParamsNum(*graph, node);

  // refresh in_tensor quant_params by out_tensor quant_params
  if (input_inited_quant_params_ < 1) {
    auto &out_tensor = graph->allTensors.at(node.outputIndex.at(0));
    auto out_quant_param = GetTensorQuantParam(out_tensor);
    MS_ASSERT(out_quant_param != nullptr);
    if (out_quant_param == nullptr || !out_quant_param->inited) {
      MS_LOG(DEBUG) << "Can not determine inputTensor quantParam from outputTensor for node " << node.name;
      return RET_ERROR;
    }
    MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
    auto &in_tensor = graph->allTensors.at(node.inputIndex.at(0));
    MS_ASSERT(in_tensor != nullptr);
    auto in_quant_param = GetTensorQuantParam(in_tensor);
    if (in_quant_param != nullptr && !in_quant_param->inited) {
      in_tensor->quantParams.front() = std::move(out_quant_param);
    }
  }

  // refresh out_tensor quant_params by in_tensor quant_params
  if (output_inited_quant_params_ < 1) {
    MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
    auto &in_tensor = graph->allTensors.at(node.inputIndex.at(0));
    MS_ASSERT(in_tensor != nullptr);
    auto in_quant_param = GetTensorQuantParam(in_tensor);
    if (in_quant_param == nullptr || !in_quant_param->inited) {
      MS_LOG(DEBUG) << "Can not determine outputTensor quantParam from inputTensor for node %s" << node.name;
      return RET_ERROR;
    }
    for (unsigned int i : node.outputIndex) {
      MS_ASSERT(graph->allTensors.size() > node.outputIndex.at(i));
      auto &out_tensor = graph->allTensors.at(i);
      MS_ASSERT(out_tensor != nullptr);
      auto out_quant_param = GetTensorQuantParam(out_tensor);
      if (out_quant_param == nullptr) {
        out_tensor->quantParams.emplace_back(std::move(in_quant_param));
        continue;
      }
      if (out_quant_param->inited) {
        continue;
      }
      out_tensor->quantParams.front() = std::move(in_quant_param);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
