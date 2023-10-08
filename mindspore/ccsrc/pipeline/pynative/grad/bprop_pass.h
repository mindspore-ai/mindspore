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
#include <utility>
#include "ir/anf.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace pynative {

namespace autograd {
class AutoGradCellImpl;
}

namespace bprop_pass {
constexpr auto kIsKNode = "is_knode";

struct PyNativePassForward {
  explicit PyNativePassForward(autograd::AutoGradCellImpl *auto_grad_cell_ptr, std::string device_target,
                               bool grad_by_value)
      : auto_grad_cell_ptr_(auto_grad_cell_ptr),
        device_target_(std::move(device_target)),
        grad_by_value_(grad_by_value) {}

  // Pass for expander outputs
  CNodePtr PassForDin(const CNodePtr &cnode, const std::string &op_name, bool is_dynamic_shape);
  // Plant op input which is tuple, and set kAttrDynInputSizes attr
  void ConvertMakeTupleInputToDynamicInput(const AnfNodePtr &node, SeenNum seen);
  // Reverse operation for pass in high grad
  void ReversePassFuncGraph(const FuncGraphPtr &func_graph);
  void ReversePassCNode(const CNodePtr &cnode, ValuePtrList *inputs_value, AnfNodePtrList *cnode_inputs);
  static inline bool need_reverse_graph() { return need_reverse_graph_; }

 private:
  CNodePtr ConvertConstInputToAttr(const CNodePtr &cnode, bool is_dynamic_shape);
  AnfNodePtr BatchNormGradToBNInferGrad(const AnfNodePtr &node, const std::string &op_name);
  void ReverseConstantToAttrNode(const CNodePtr &cnode, ValuePtrList *inputs_value, AnfNodePtrList *cnode_inputs);
  void ReverseMakeTupleNode(const CNodePtr &cnode, ValuePtrList *inputs_value, AnfNodePtrList *cnode_inputs);
  void ReverseBNInfer(const CNodePtr &cnode);
  void ReverseCNodeInputs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs, ValuePtrList *inputs_value);

  autograd::AutoGradCellImpl *auto_grad_cell_ptr_{nullptr};
  std::string device_target_;
  bool grad_by_value_{false};
  static bool need_reverse_graph_;
};

void ClearCache();
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_PASS_H_
