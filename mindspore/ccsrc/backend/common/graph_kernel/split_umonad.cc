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
#include "backend/common/graph_kernel/split_umonad.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_info.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore::graphkernel {
const BaseRef SplitAssign::DefinePattern() const {
  VarPtr v = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<Var>();
  VarPtr Us = std::make_shared<Var>();
  VarPtr UMonad = std::make_shared<Var>();
  return VectorRef({v, Xs, Us, UMonad});
}

bool CanSplit(const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimAssign); }

AnfNodePtr ProcessNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // Get original op's abstract and inputs
  AbstractBasePtr original_abstract = cnode->abstract()->Clone();
  auto original_inputs = cnode->inputs();

  // Create depend node
  AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), original_inputs[input_idx], original_inputs.back()};
  auto depend_cnode = func_graph->NewCNode(depend_inputs);
  depend_cnode->set_abstract(original_inputs[input_idx]->abstract());
  depend_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  // Create new node, delete U from inputs.
  AnfNodePtrList new_inputs = {cnode->input(0)};
  for (size_t i = 1; i + 1 < cnode->size(); i++) {
    if (i == input_idx) {
      new_inputs.push_back(depend_cnode);
    } else {
      new_inputs.push_back(cnode->input(i));
    }
  }
  auto new_cnode = func_graph->NewCNode(new_inputs);
  new_cnode->set_abstract(original_abstract);
  new_cnode->set_kernel_info(cnode->kernel_info_ptr());
  return new_cnode;
}

const AnfNodePtr SplitAssign::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!CanSplit(node)) {
    return node;
  }
  return ProcessNode(node->func_graph(), node, 1);
}

AnfNodePtr OpUMonadExpanderDeco::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = node;
  // assume the UMonad node is the last input
  if (cnode->size() > 1 && HasAbstractUMonad(cnode->inputs().back())) {
    new_node = ProcessNode(node->func_graph(), node, input_idx_);
  }
  return decorated_->Run(new_node);
}
}  // namespace mindspore::graphkernel
