/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "pre_activate/ascend/format_type/remove_no_use_reshape_op.h"
#include <vector>
#include <memory>
#include "pre_activate/common/helper.h"
#include "kernel/common_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr RemoveReshapeOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (op_name != prim::kPrimReshape->name()) {
    return nullptr;
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto input_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, 0);
  if (input_shape.size() != 1 || input_format != kOpFormat_NC1HWC0) {
    return nullptr;
  }

  return cnode->input(1);
}
}  // namespace

const AnfNodePtr RemoveNoUseReshapeOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsGraphKernel(node)) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "====process op: " << AnfAlgo::GetCNodeName(node);
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> todos;
  kernel::GetValidKernelNodes(fg, &todos);
  for (auto &t : todos) {
    auto new_node = RemoveReshapeOp(t->cast<CNodePtr>());
    if (new_node != nullptr && new_node != t) {
      (void)manager->Replace(t, new_node);
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
