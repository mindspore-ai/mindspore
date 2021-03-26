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
#include "tools/converter/quantizer/quant_helper/conv_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/conv_quant_param_propogator.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/ir/dtype/type_id.h"
namespace mindspore::lite {
static constexpr size_t kInputIndex = 0;
static constexpr size_t kWeightIndex = 1;

bool ConvQuantTypeDeterminer::DetermineQuantWeight(const mindspore::schema::MetaGraphT &graph,
                                                   mindspore::schema::CNodeT *node) {
  MS_ASSERT(node->inputIndex.size() >= 2);
  auto &input_tensor = graph.allTensors.at(node->inputIndex.at(kInputIndex));
  auto &weight_tensor = graph.allTensors.at(node->inputIndex.at(kWeightIndex));
  if (input_tensor->quantParams.empty() || !input_tensor->quantParams.front()->inited) {
    if (!weight_tensor->quantParams.empty() && weight_tensor->quantParams.front()->inited) {
      node->quantType = schema::QuantType_QUANT_WEIGHT;
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::lite
