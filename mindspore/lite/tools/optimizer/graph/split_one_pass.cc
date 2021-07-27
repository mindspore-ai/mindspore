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

#include "tools/optimizer/graph/split_one_pass.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "ops/split.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMinCnodeSize = 2;
}  // namespace
bool SplitOnePass::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  // this pass handle this: split with split num 1
  // after this pass, such split op will be removed.
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimSplit)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      return false;
    }
    auto primitive_c = GetValueNode<std::shared_ptr<mindspore::ops::Split>>(cnode->input(0));
    if (primitive_c == nullptr) {
      return false;
    }
    if (primitive_c->get_output_num() != 1) {
      continue;
    }
    if (cnode->size() < kMinCnodeSize) {
      return false;
    }
    func_graph->manager()->Replace(node, cnode->input(1));
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
