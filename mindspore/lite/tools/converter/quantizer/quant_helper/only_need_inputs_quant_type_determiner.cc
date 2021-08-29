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
#include "tools/converter/quantizer/quant_helper/only_need_inputs_quant_type_determiner.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore::lite {
bool OnlyNeedInputsQuantTypeDeterminer::DetermineQuantAll(const schema::MetaGraphT &graph, schema::CNodeT *node) {
  MS_ASSERT(node != nullptr);
  UpdateQuantParamsNum(graph, *node);
  if (input_inited_quant_params_ == node->inputIndex.size()) {
    node->quantType = schema::QuantType_QUANT_ALL;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
