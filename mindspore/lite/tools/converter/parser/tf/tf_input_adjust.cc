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

#include "tools/converter/parser/tf/tf_input_adjust.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace lite {
namespace {
STATUS ReplaceConstant(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() != opt::kInputSizeTwo) {
    MS_LOG(ERROR) << "TF's constant-op must have two inputs, but got " << cnode->size();
    return RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (!manager->Replace(cnode, cnode->input(1))) {
    MS_LOG(ERROR) << "Replace old-node with manager failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

bool TfInputAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!opt::CheckPrimitiveType(node, prim::kPrimConstant)) {
      continue;
    }
    status = ReplaceConstant(func_graph, cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Adjust TF constant-op failed.";
      return false;
    }
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
