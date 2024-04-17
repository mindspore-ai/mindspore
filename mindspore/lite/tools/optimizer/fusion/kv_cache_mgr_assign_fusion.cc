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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/kv_cache_mgr_assign_fusion.h"
#include <vector>
#include <memory>
#include "ops/array_ops.h"
#include "ops/math_ops.h"
#include "ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/nn_optimizer_ops.h"

namespace mindspore {
namespace opt {
int KVCacheMgrAssignFusion::RemoveAssignOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager,
                                           const CNodePtr &kv_cache_cnode) {
  const int expected_assign_input_count = 4;
  auto assign_cnode = anf_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(assign_cnode->size() == expected_assign_input_count, lite::RET_NO_CHANGE);
  const int past_input_index = 1;
  const int kv_cache_mgr_input_index = 2;
  if (kv_cache_cnode->input(past_input_index) != assign_cnode->input(past_input_index)) {
    MS_LOG(INFO) << "kv_cache_cnode->input(1) != assign_cnode->input(1)";
    return lite::RET_NO_CHANGE;
  }
  if (kv_cache_cnode != assign_cnode->input(kv_cache_mgr_input_index)) {
    MS_LOG(INFO) << "kv_cache_cnode != assign_cnode->input(2)";
    return lite::RET_NO_CHANGE;
  }
  (void)this->remove_cnode_.insert(anf_node);
  return manager->Replace(anf_node, kv_cache_cnode) ? RET_OK : RET_ERROR;
}

bool KVCacheMgrAssignFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
    func_graph->set_manager(manager);
  }
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto node_list = TopoSort(func_graph->get_return());
  CNodePtr first_concat_cnode = nullptr;
  AnfNodePtr batch_valid_length = nullptr;
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimKVCacheMgr)) {
      continue;
    }
    auto kv_cache_cnode = node->cast<CNodePtr>();
    auto iter = manager->node_users().find(kv_cache_cnode);
    if (iter == manager->node_users().end()) {
      MS_LOG(ERROR) << "node has no output in manager";
      return false;
    }
    auto output_list = iter->second;
    for (auto &out : output_list) {
      if (!utils::isa<CNodePtr>(out.first)) {
        continue;
      }
      auto status = lite::RET_OK;
      if (CheckPrimitiveType(out.first, prim::kPrimAssign)) {
        status = this->RemoveAssignOp(out.first, manager, kv_cache_cnode);
      }
      if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
        MS_LOG(ERROR) << "Failed to run kv_cache_mgr assign elimination pass.";
        return false;
      }
    }
    for (auto &drop_node : this->remove_cnode_) {
      func_graph->DropNode(drop_node);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
