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
#include "tools/converter/quantizer/quant_helper/carry_data_quant_type_determiner.h"
#include <utility>
#include <memory>
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"
namespace mindspore::lite {
bool CarryDataQuantTypeDeterminer::DetermineQuantAll(const schema::MetaGraphT &graph, schema::CNodeT *node) {
  MS_ASSERT(node->inputIndex.size() >= 1);
  MS_ASSERT(node->outputIndex.size() >= 1);
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");

  // check first in tensor
  MS_ASSERT(graph.allTensors.size() > node->inputIndex.at(0));
  auto &in_tensor = graph.allTensors.at(node->inputIndex.at(0));
  if (!in_tensor->quantParams.empty()) {
    if (std::any_of(in_tensor->quantParams.begin(), in_tensor->quantParams.end(),
                    [](const std::unique_ptr<QuantParamT> &quant_param) { return !quant_param->inited; })) {
      return false;
    }
  } else {
    return false;
  }

  // check first out tensor
  MS_ASSERT(graph.allTensors.size() > node->outputIndex.at(0));
  auto &out_tensor = graph.allTensors.at(node->outputIndex.at(0));
  if (!out_tensor->quantParams.empty()) {
    if (std::any_of(out_tensor->quantParams.begin(), out_tensor->quantParams.end(),
                    [](const std::unique_ptr<QuantParamT> &quant_param) { return !quant_param->inited; })) {
      return false;
    }
    node->quantType = schema::QuantType_QUANT_ALL;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
