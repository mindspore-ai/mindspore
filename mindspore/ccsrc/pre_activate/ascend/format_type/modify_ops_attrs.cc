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

#include "pre_activate/ascend/format_type/modify_ops_attrs.h"
#include <vector>
#include <memory>
#include "utils/utils.h"
#include "pre_activate/common/helper.h"
#include "kernel/common_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr ModifyReduceOpsAttrs(const CNodePtr &cnode) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto input_format = AnfAlgo::GetInputFormat(cnode, 0);
  if (input_shape.size() == 5 || input_format != kOpFormat_NC1HWC0) {
    return nullptr;
  }
  if (!AnfAlgo::HasNodeAttr(kAttrKeepDims, cnode)) {
    return nullptr;
  }

  AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(true), cnode);
  return cnode;
}

AnfNodePtr ModifyTileOpAttrs(const CNodePtr &cnode) {
  auto input_shape = AnfAlgo::GetInputDeviceShape(cnode, 0);
  if (input_shape.size() != 5) {
    return nullptr;
  }
  if (!AnfAlgo::HasNodeAttr(kAttrMultiples, cnode)) {
    return nullptr;
  }

  auto multiples = AnfAlgo::GetNodeAttr<std::vector<int>>(cnode, kAttrMultiples);
  if (multiples.size() == 4 && multiples[1] == 1) {
    multiples.push_back(1);
    AnfAlgo::SetNodeAttr(kAttrMultiples, MakeValue(multiples), cnode);
  }

  return cnode;
}

AnfNodePtr ModifyAttrs(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (op_name == prim::kPrimTile->name()) {
    return ModifyTileOpAttrs(cnode);
  } else if (op_name == prim::kPrimReduceSum->name()) {
    // kPrimReduceMean
    // kPrimReduceSum
    // kPrimReduceAll
    // kPrimReduceMax
    // kPrimReduceMin
    return ModifyReduceOpsAttrs(cnode);
  }
  return nullptr;
}
}  // namespace

const AnfNodePtr ModifyOpAttrs::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsGraphKernel(node)) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "====Process op: " << AnfAlgo::GetCNodeName(node);
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> todos;
  kernel::GetValidKernelNodes(fg, &todos);
  for (auto &t : todos) {
    auto new_node = ModifyAttrs(t->cast<CNodePtr>());
    if (new_node != nullptr && new_node != t) {
      (void)manager->Replace(t, new_node);
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
