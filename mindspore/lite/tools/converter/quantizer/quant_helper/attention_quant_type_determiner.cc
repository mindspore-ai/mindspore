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
#include "tools/converter/quantizer/quant_helper/attention_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/conv_quant_param_propogator.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
const size_t kWeightQueryIndex = 4;
const size_t kWeightKeyIndex = 5;
const size_t kWeightValueIndex = 6;
const size_t kWeightOutputIndex = 10;

bool AttentionQuantTypeDeterminer::DetermineQuantWeight(const mindspore::schema::MetaGraphT &graph,
                                                        mindspore::schema::CNodeT *node) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  auto input_index = node->inputIndex;
  MS_CHECK_FALSE_MSG(input_index.empty(), false, "inputIndex is empty.");
  MS_CHECK_TRUE_MSG(input_index.size() > kInputIndex, false, "invalid access.");
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > input_index.at(kInputIndex), false, "invalid access.");
  auto &input_tensor = graph.allTensors.at(input_index.at(kInputIndex));

  MS_CHECK_TRUE_MSG(input_index.size() > kWeightQueryIndex, false, "invalid access.");
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > input_index.at(kWeightQueryIndex), false, "invalid access.");
  auto &weight_query_tensor = graph.allTensors.at(input_index.at(kWeightQueryIndex));

  MS_CHECK_TRUE_MSG(input_index.size() > kWeightKeyIndex, false, "invalid access.");
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > input_index.at(kWeightKeyIndex), false, "invalid access.");
  auto &weight_key_tensor = graph.allTensors.at(input_index.at(kWeightKeyIndex));

  MS_CHECK_TRUE_MSG(input_index.size() > kWeightValueIndex, false, "invalid access.");
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > input_index.at(kWeightValueIndex), false, "invalid access.");
  auto &weight_value_tensor = graph.allTensors.at(input_index.at(kWeightValueIndex));

  MS_CHECK_TRUE_MSG(input_index.size() > kWeightOutputIndex, false, "invalid access.");
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > input_index.at(kWeightOutputIndex), false, "invalid access.");
  auto &weight_output_tensor = graph.allTensors.at(input_index.at(kWeightOutputIndex));

  MS_CHECK_TRUE_RET(input_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(weight_query_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(weight_key_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(weight_value_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(weight_output_tensor != nullptr, false);
  if (!quant::TensorQuantParamsInited(*input_tensor) && quant::TensorQuantParamsInited(*weight_query_tensor) &&
      quant::TensorQuantParamsInited(*weight_key_tensor) && quant::TensorQuantParamsInited(*weight_value_tensor) &&
      quant::TensorQuantParamsInited(*weight_output_tensor)) {
    node->quantType = schema::QuantType_QUANT_WEIGHT;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
