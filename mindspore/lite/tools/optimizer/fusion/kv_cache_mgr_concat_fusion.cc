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
#include "tools/optimizer/fusion/kv_cache_mgr_concat_fusion.h"
#include <vector>
#include <memory>
#include "ops/array_ops.h"
#include "ops/math_ops.h"
#include "ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
AnfNodePtr KVCacheMgrConcatFusion::GetBatchValidLength(CNodePtr concat_cnode) {
  auto make_tuple_node = concat_cnode->input(kInputIndexOne);
  MS_CHECK_TRUE_RET(make_tuple_node != nullptr, nullptr);
  auto make_tuple_cnode = make_tuple_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(make_tuple_cnode != nullptr, nullptr);
  const size_t kMakeTupleInputNum = 3;
  if (make_tuple_cnode->inputs().size() != kMakeTupleInputNum) {
    return nullptr;
  }

  return make_tuple_cnode->input(kInputIndexOne);
}

bool KVCacheMgrConcatFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  bool is_fisrt_concat = true;
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
    MS_CHECK_TRUE_RET(kv_cache_cnode != nullptr, false);
    auto concat_cnode = kv_cache_cnode->input(kInputIndexThree)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(concat_cnode != nullptr, false);
    if (!CheckPrimitiveType(concat_cnode, prim::kPrimConcat)) {
      return false;
    }
    if (is_fisrt_concat) {
      is_fisrt_concat = false;
      first_concat_cnode = concat_cnode;
      batch_valid_length = GetBatchValidLength(first_concat_cnode);
      continue;
    }
    if (GetBatchValidLength(concat_cnode) != batch_valid_length) {
      continue;
    }

    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
      func_graph->set_manager(manager);
    }
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    kv_cache_cnode->set_input(kInputIndexThree, first_concat_cnode);

    (void)manager->Replace(concat_cnode, first_concat_cnode);
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
