/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "src//common/log_util.h"
#include "nnacl/op_base.h"
namespace mindspore::lite {
int CarryDataQuantParamPropogator::PropogateQuantParams(schema::MetaGraphT *graph, const schema::CNodeT &node) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "Graph is nullptr.");
  UpdateQuantParamsNum(*graph, node);

  MS_CHECK_FALSE_MSG(graph->allTensors.empty(), RET_ERROR, "Tensors is empty.");
  // refresh in_tensor quant_params by out_tensor quant_params
  if (input_inited_quant_params_ < 1) {
    MS_CHECK_FALSE_MSG(node.outputIndex.empty(), RET_ERROR, "OutputIndex is empty.");
    MS_CHECK_TRUE_RET(graph->allTensors.size() > node.outputIndex.at(0), RET_ERROR);
    auto &out_tensor = graph->allTensors.at(node.outputIndex.at(0));
    auto out_quant_param = GetTensorQuantParam(out_tensor);
    if (out_quant_param == nullptr || !out_quant_param->inited) {
      MS_LOG(DEBUG) << node.name << " dont need to pass quant param.";
      return RET_NO_CHANGE;
    }
    MS_CHECK_FALSE_MSG(node.inputIndex.empty(), RET_ERROR, "inputIndex is empty.");
    MS_CHECK_TRUE_RET(graph->allTensors.size() > node.inputIndex.at(0), RET_ERROR);
    auto &in_tensor = graph->allTensors.at(node.inputIndex.at(0));
    MS_CHECK_TRUE_RET(in_tensor != nullptr, RET_NULL_PTR);
    auto in_quant_param = GetTensorQuantParam(in_tensor);
    if (in_quant_param != nullptr && !in_quant_param->inited) {
      MS_CHECK_FALSE_MSG(in_tensor->quantParams.empty(), RET_ERROR, "in_tensor quantParams is empty.");
      in_tensor->quantParams.front() = std::move(out_quant_param);
    }
  }

  // refresh out_tensor quant_params by in_tensor quant_params
  if (output_inited_quant_params_ < 1) {
    MS_CHECK_FALSE_MSG(node.inputIndex.empty(), RET_ERROR, "inputIndex is empty.");
    MS_CHECK_TRUE_RET(graph->allTensors.size() > node.inputIndex.at(0), RET_ERROR);
    auto &in_tensor = graph->allTensors.at(node.inputIndex.at(0));
    MS_CHECK_TRUE_RET(in_tensor != nullptr, RET_NULL_PTR);
    auto in_quant_param = GetTensorQuantParam(in_tensor);
    if (in_quant_param == nullptr || !in_quant_param->inited) {
      MS_LOG(DEBUG) << node.name << " dont need to pass quant param.";
      return RET_NO_CHANGE;
    }
    for (unsigned int i : node.outputIndex) {
      MS_CHECK_TRUE_RET(graph->allTensors.size() > i, RET_ERROR);
      auto &out_tensor = graph->allTensors.at(i);
      MS_CHECK_TRUE_RET(out_tensor != nullptr, RET_NULL_PTR);
      auto out_quant_param = GetTensorQuantParam(out_tensor);
      if (out_quant_param == nullptr) {
        out_tensor->quantParams.emplace_back(std::move(in_quant_param));
        continue;
      }
      if (out_quant_param->inited) {
        continue;
      }
      MS_CHECK_FALSE_MSG(out_tensor->quantParams.empty(), RET_ERROR, "out_tensor quantParams is empty.");
      out_tensor->quantParams.front() = std::move(in_quant_param);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
