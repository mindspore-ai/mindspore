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

#include "backend/optimizer/ascend/format_type/add_attr_for_3d_graph.h"
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
void AddAttrForAllCNode(const std::vector<AnfNodePtr> &node_list) {
  for (auto node : node_list) {
    if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
      continue;
    }
    AnfAlgo::SetNodeAttr("io_format", MakeValue(kOpFormat_NCDHW), node);
  }
}

bool NodeHasAttrIoFormat(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::HasNodeAttr("io_format", cnode)) {
      auto attr = AnfAlgo::GetNodeAttr<std::string>(cnode, "io_format");
      return attr == kOpFormat_NCDHW;
    }
  }
  return false;
}
}  // namespace

bool AddIoFormatAttrFor3DGraph::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  bool changed = false;
  if (std::any_of(node_list.begin(), node_list.end(),
                  [](const AnfNodePtr &node) { return NodeHasAttrIoFormat(node); })) {
    AddAttrForAllCNode(node_list);
    changed = true;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
