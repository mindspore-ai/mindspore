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
#include "tools/converter/quantizer/quant_helper/conv_quant_type_determiner.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
namespace mindspore::lite {
bool ConvQuantTypeDeterminer::DetermineQuantWeight(const mindspore::schema::MetaGraphT &graph,
                                                   mindspore::schema::CNodeT *node) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  MS_CHECK_TRUE_RET(node->inputIndex.size() >= kInputIndexTwo, false);
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kInputIndex), false, "Out of vector range.");
  auto &input_tensor = graph.allTensors.at(node->inputIndex.at(kInputIndex));
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kWeightIndex), false, "Out of vector range.");
  auto &weight_tensor = graph.allTensors.at(node->inputIndex.at(kWeightIndex));
  MS_CHECK_TRUE_RET(node->outputIndex.size() > kOutputIndex, false);
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->outputIndex.at(kOutputIndex), false, "Out of vector range.");
  auto &output_tensor = graph.allTensors.at(node->outputIndex.at(kOutputIndex));

  MS_CHECK_TRUE_RET(input_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(output_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(weight_tensor != nullptr, false);
  if ((!quant::TensorQuantParamsInited(*input_tensor) || !quant::TensorQuantParamsInited(*output_tensor)) &&
      quant::TensorQuantParamsInited(*weight_tensor)) {
    node->quantType = schema::QuantType_QUANT_WEIGHT;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
