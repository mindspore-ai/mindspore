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
#include "tools/optimizer/fusion/kv_cache_mgr_load_fusion.h"
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
int KVCacheMgrLoadFusion::RemoveLoadOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager,
                                       const CNodePtr &kv_cache_cnode) {
  const int expected_load_input_count = 3;
  auto load_cnode = anf_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(load_cnode->inputs().size() == expected_load_input_count, lite::RET_NO_CHANGE);
  const size_t past_input_index = 1;
  const size_t input_para_index = 1;
  if (kv_cache_cnode->input(past_input_index) != load_cnode) {
    MS_LOG(INFO) << "kv_cache_cnode->input(1) != load_cnode";
    return lite::RET_NO_CHANGE;
  }
  auto past_para_ptr = load_cnode->input(input_para_index);
  if (!utils::isa<ParameterPtr>(past_para_ptr)) {
    MS_LOG(INFO) << "load_cnode input is not parameter";
    return lite::RET_NO_CHANGE;
  }

  (void)this->remove_cnode_.insert(anf_node);
  kv_cache_cnode->set_input(past_input_index, past_para_ptr);
  return manager->Replace(anf_node, past_para_ptr) ? RET_OK : RET_ERROR;
}

bool KVCacheMgrLoadFusion::Run(const FuncGraphPtr &func_graph) {
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
    for (auto &input : kv_cache_cnode->inputs()) {
      if (!utils::isa<CNodePtr>(input)) {
        continue;
      }
      auto status = lite::RET_OK;
      if (CheckPrimitiveType(input, prim::kPrimLoad)) {
        status = this->RemoveLoadOp(input, manager, kv_cache_cnode);
      }
      if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
        MS_LOG(ERROR) << "Failed to run kv_cache_mgr load elimination pass.";
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
