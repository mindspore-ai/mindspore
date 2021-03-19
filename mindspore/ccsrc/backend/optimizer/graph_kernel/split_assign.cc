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
#include "backend/optimizer/graph_kernel/split_assign.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "base/core_ops.h"
#include "utils/utils.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef SplitAssign::DefinePattern() const {
  VarPtr v = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<Var>();
  VarPtr Us = std::make_shared<Var>();
  VarPtr UMonad = std::make_shared<Var>();
  return VectorRef({v, Xs, Us, UMonad});
}

bool CanSplit(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimAssignAdd) || IsPrimitiveCNode(node, prim::kPrimAssign) ||
         IsPrimitiveCNode(node, prim::kPrimAssignSub);
}

const AnfNodePtr SplitAssign::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!CanSplit(node)) return node;
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kAssignInputTensorNum);
  // Get original assign op's abstract and inputs
  AbstractBasePtr original_abstract = cnode->abstract()->Clone();
  auto original_inputs = cnode->inputs();
  // Create depend node
  AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), original_inputs[1], original_inputs[3]};
  auto depend_cnode = func_graph->NewCNode(depend_inputs);
  depend_cnode->set_abstract(original_inputs[1]->abstract());
  depend_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  // Create new assign node, delete U from inputs.
  AnfNodePtrList new_assign_inputs = {cnode->input(0), depend_cnode, original_inputs[2]};
  auto new_assign_cnode = func_graph->NewCNode(new_assign_inputs);
  new_assign_cnode->set_abstract(original_abstract);
  new_assign_cnode->set_kernel_info(cnode->kernel_info_ptr());
  return new_assign_cnode;
}
}  // namespace opt
}  // namespace mindspore
