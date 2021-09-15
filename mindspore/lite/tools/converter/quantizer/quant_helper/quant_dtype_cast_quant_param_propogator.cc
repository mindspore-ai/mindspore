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
#include "tools/converter/quantizer/quant_helper/quant_dtype_cast_quant_param_propogator.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
STATUS QuantDtypeCastQuantParamPropogator::PropogateQuantParams(schema::MetaGraphT *graph, const schema::CNodeT &node) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(!node.inputIndex.empty(), RET_ERROR, "inputIndex is empty.");
  MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
  auto &input_tensor = graph->allTensors.at(node.inputIndex.at(0));
  if (!input_tensor->quantParams.empty() && input_tensor->quantParams.front()->inited) {
    input_tensor->quantParams.front()->dstDtype = input_tensor->dataType;
  }
  MS_ASSERT(graph->allTensors.size() > node.outputIndex.at(0));
  auto &output_tensor = graph->allTensors.at(node.outputIndex.at(0));
  if (!output_tensor->quantParams.empty() && output_tensor->quantParams.front()->inited) {
    output_tensor->quantParams.front()->dstDtype = output_tensor->dataType;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
