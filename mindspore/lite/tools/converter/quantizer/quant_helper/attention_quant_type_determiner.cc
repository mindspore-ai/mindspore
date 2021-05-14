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
#include "tools/converter/quantizer/quant_helper/attention_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/conv_quant_param_propogator.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/ir/dtype/type_id.h"
namespace mindspore::lite {
bool AttentionQuantTypeDeterminer::DetermineQuantWeight(const mindspore::schema::MetaGraphT &graph,
                                                        mindspore::schema::CNodeT *node) {
  MS_ASSERT(node->inputIndex.size() >= 2);
  auto &input_tensor = graph.allTensors.at(node->inputIndex.at(kInputIndex));
  auto &weight_query_tensor = graph.allTensors.at(node->inputIndex.at(4));
  auto &weight_key_tensor = graph.allTensors.at(node->inputIndex.at(5));
  auto &weight_value_tensor = graph.allTensors.at(node->inputIndex.at(6));
  auto &weight_output_tensor = graph.allTensors.at(node->inputIndex.at(10));

  if (!quant::TensorQuantParamsInited(*input_tensor) && quant::TensorQuantParamsInited(*weight_query_tensor) &&
      quant::TensorQuantParamsInited(*weight_key_tensor) && quant::TensorQuantParamsInited(*weight_value_tensor) &&
      quant::TensorQuantParamsInited(*weight_output_tensor)) {
    node->quantType = schema::QuantType_QUANT_WEIGHT;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
