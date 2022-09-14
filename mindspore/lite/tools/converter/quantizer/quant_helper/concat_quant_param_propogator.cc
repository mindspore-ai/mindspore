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
#include "tools/converter/quantizer/quant_helper/concat_quant_param_propogator.h"
#include <cfloat>
#include <memory>
#include <utility>
#include "src/common/log_adapter.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
int ConcatQuantParamPropogator::PropogateQuantParams(mindspore::schema::MetaGraphT *graph,
                                                     const mindspore::schema::CNodeT &node) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  UpdateQuantParamsNum(*graph, node);

  if (input_inited_quant_params_ != node.inputIndex.size()) {
    MS_LOG(DEBUG) << "Can not determine concat inputTensor quantParam, node " << node.name;
    return RET_NO_CHANGE;
  }

  if (output_inited_quant_params_ != 1) {
    MS_CHECK_TRUE_RET(output_inited_quant_params_ == 0, RET_ERROR);
    float min_min = FLT_MAX;
    float max_max = FLT_MIN;
    bool narrow_range = false;
    int num_bits = -1;
    for (size_t index : node.inputIndex) {
      MS_ASSERT(graph->allTensors.size() > index);
      auto &in_tensor = graph->allTensors.at(index);
      MS_ASSERT(in_tensor != nullptr);
      auto in_quant_param = GetTensorQuantParam(in_tensor);
      if (in_quant_param == nullptr || !in_quant_param->inited) {
        return RET_ERROR;
      }
      if (num_bits == -1) {
        narrow_range = in_quant_param->narrowRange;
        num_bits = in_quant_param->numBits;
      } else {
        MS_ASSERT(narrow_range == quantParam->narrowRange);
        MS_ASSERT(num_bits == quantParam->numBits);
      }

      if (in_quant_param->max < in_quant_param->min) {
        MS_LOG(DEBUG) << "Input quant param is invalid for propogator";
        return RET_ERROR;
      }

      if (min_min > in_quant_param->min) {
        min_min = in_quant_param->min;
      }
      if (max_max < in_quant_param->max) {
        max_max = in_quant_param->max;
      }
    }

    MS_CHECK_FALSE_MSG(node.outputIndex.empty(), RET_ERROR, "Output index is empty.");
    MS_CHECK_TRUE_RET(graph->allTensors.size() > node.outputIndex.front(), RET_ERROR);
    auto &out_tensor = graph->allTensors.at(node.outputIndex.front());
    MS_CHECK_TRUE_RET(out_tensor != nullptr, RET_NULL_PTR);
    auto out_quant_param = std::make_unique<QuantParamT>();
    MS_CHECK_TRUE_MSG(out_quant_param != nullptr, RET_NULL_PTR, "out_quant_param is nullptr.");

    auto status = CalQuantizationParams(out_quant_param.get(), min_min, max_max, num_bits, narrow_range);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
      return RET_ERROR;
    }
    out_tensor->quantParams.emplace_back(std::move(out_quant_param));
    output_inited_quant_params_++;
  }

  return RET_OK;
}
}  // namespace mindspore::lite
