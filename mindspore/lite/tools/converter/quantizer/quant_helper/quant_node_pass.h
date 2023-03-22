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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QUANT_NODE_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QUANT_NODE_PASS_H_

#include <string>
#include <map>
#include <vector>
#include <set>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ops/primitive_c.h"
#include "ops/tuple_get_item.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
class QuantNodePass {
 public:
  explicit QuantNodePass(const FuncGraphPtr &func_graph) : func_graph_(func_graph) {}

  ~QuantNodePass() = default;

  int Quant();

 private:
  int DoWeightQuant(const CNodePtr &cnode);
  int QuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                  const std::vector<schema::QuantParamT> &quant_params, int preferred_dim);
  int DoFullQuant(const CNodePtr &cnode);
  int DoParameterNodeQuant(const CNodePtr &cnode, const ParameterPtr &input_node, size_t input_index);
  int DoValueNodeQuant(const CNodePtr &cnode, const ValueNodePtr &input_node, size_t input_index);
  int CheckNodeDType(const CNodePtr &cnode, const AnfNodePtr &input_node, size_t input_index);
  bool CanTensorQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node);

 private:
  FuncGraphPtr func_graph_ = nullptr;
  // key is tensor_name, to delete
  std::map<std::string, std::vector<schema::QuantParamT>> weight_quant_params_bak_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QUANT_NODE_PASS_H_
