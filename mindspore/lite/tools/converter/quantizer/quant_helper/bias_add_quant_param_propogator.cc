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

#define USE_DEPRECATED_API
#include "tools/converter/quantizer/quant_helper/bias_add_quant_param_propogator.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
static constexpr size_t kBiasAddSize = 2;
int BiasAddQuantParamPropogator::PropogateQuantParams(mindspore::schema::MetaGraphT *graph,
                                                      const mindspore::schema::CNodeT &node) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  if (node.inputIndex.size() == kBiasAddSize) {
    auto &bias_tensor = graph->allTensors.at(node.inputIndex.at(kBiasAddSize - 1));
    MS_CHECK_TRUE_RET(bias_tensor != nullptr, RET_NULL_PTR);
    for (auto &quantParam : bias_tensor->quantParams) {
      quantParam->dstDtype = TypeId::kNumberTypeInt32;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
