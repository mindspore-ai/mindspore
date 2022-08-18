/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_DTYPE_TRANSFORM_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_DTYPE_TRANSFORM_PASS_H_

#include <memory>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ops/primitive_c.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_params.h"

namespace mindspore::lite::quant {
/**
 * Transform CNode(dtype,uint8toint8,weigh data)
 * Insert QuantCastNode
 * */
class DTypeTransformPass {
 public:
  explicit DTypeTransformPass(const FuncGraphPtr &func_graph) : func_graph_(func_graph) {}

  ~DTypeTransformPass() = default;

  int Transform();

 private:
  int DoParameterNodeTrans(const CNodePtr &cnode, const ParameterPtr &input_node, size_t input_index);

  int Uint8toInt8(uint8_t *data, int size);

  int GetQuantType(const CNodePtr &cnode, schema::QuantType *quant_type);

  int DoNodeDTypeTrans(const CNodePtr &cnode);

  bool CheckNeedDTypeTrans(const CNodePtr &cnode);

  int InsertForwardCastNode(const CNodePtr &cnode, schema::QuantType curr_quant_type);

  int InsertBackwardCastNode(const CNodePtr &cnode, schema::QuantType curr_quant_type);

  FuncGraphPtr func_graph_ = nullptr;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_DTYPE_TRANSFORM_PASS_H_
