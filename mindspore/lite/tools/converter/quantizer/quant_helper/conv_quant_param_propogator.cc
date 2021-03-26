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
namespace mindspore::lite {
static constexpr size_t kBiasAdd = 3;

STATUS ConvQuantParamPropogator::PropogateQuantParams(mindspore::schema::MetaGraphT *graph,
                                                      const mindspore::schema::CNodeT &node) {
  if (node.inputIndex.size() == kBiasAdd) {
    auto &bias_tensor = graph->allTensors.at(node.inputIndex.at(kBiasAdd - 1));
    for (auto &quantParam : bias_tensor->quantParams) {
      quantParam->dstDtype = TypeId::kNumberTypeInt32;
    }
  }
  return RET_OK;
}

}  // namespace mindspore::lite
