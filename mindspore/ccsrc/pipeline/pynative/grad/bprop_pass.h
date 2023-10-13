/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_PASS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_PASS_H_

#include <string>
#include "ir/anf.h"

namespace mindspore {
namespace pynative {

namespace autograd {
class AutoGradCellImpl;
}

namespace bprop_pass {
constexpr auto kIsKNode = "is_knode";

void ConvertValueNodeValueToTensor(const AnfNodePtr &din);
void ConvertMakeTupleInputToDynamicInput(const AnfNodePtr &node, SeenNum seen,
                                         autograd::AutoGradCellImpl *auto_grad_cell_ptr);
CNodePtr ConvertConstInputToAttr(const CNodePtr &cnode, const std::string &device_target, bool is_dynamic_shape,
                                 bool grad_by_value);
void ProcessAttrNode(const FuncGraphPtr &tape_graph, const CNodePtr &cnode, ValuePtrList *input_value,
                     AnfNodePtrList *cnode_inputs);
void ClearCache();
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_PASS_H_
