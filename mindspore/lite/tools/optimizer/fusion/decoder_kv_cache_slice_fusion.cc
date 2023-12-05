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
#include "tools/optimizer/fusion/decoder_kv_cache_slice_fusion.h"
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
int DecoderKVCacheSliceFusion::RemoveSliceOp(const FuncGraphManagerPtr &manager, const AnfNodePtr &slice_anf_node,
                                             const AnfNodePtr &load_anf_node) {
  MS_LOG(INFO) << "Start remove slice op.";

  auto load_node = load_anf_node->cast<CNodePtr>();
  if (!manager->Replace(slice_anf_node, load_node)) {
    MS_LOG(ERROR) << "replace strided slice with load node";
    return RET_ERROR;
  }

  MS_LOG(INFO) << "add remove node ：" << slice_anf_node->fullname_with_scope();
  (void)this->remove_cnode_.insert(slice_anf_node);
  return RET_OK;
}

std::shared_ptr<mindspore::AnfNode> FindPrimCnode(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                                  const PrimitivePtr &primitive_type) {
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);

  auto node = anf_node->cast<CNodePtr>();
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(ERROR) << "node has no output in manager";
    return nullptr;
  }
  auto output_list = iter->second;
  for (auto &out : output_list) {
    if (!utils::isa<CNodePtr>(out.first)) {
      continue;
    }
    auto out_node = out.first;
    if (CheckPrimitiveType(out_node, primitive_type)) {
      MS_LOG(INFO) << "Find " << primitive_type->ToString() << " : " << node->fullname_with_scope();
      return out_node;
    }
  }
  return nullptr;
}

bool DecoderKVCacheSliceFusion::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "decoder kv cache and slice fusion run start.";

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

    if (!CheckPrimitiveType(node, prim::kPrimDecoderKVCache)) {
      continue;
    }
    MS_LOG(INFO) << "Find decoder kv cache node : " << node->fullname_with_scope();

    auto update_state_anf_node = FindPrimCnode(func_graph, node, prim::kPrimUpdateState);
    if (update_state_anf_node == nullptr) {
      continue;
    }

    auto load_anf_node = FindPrimCnode(func_graph, update_state_anf_node, prim::kPrimLoad);
    if (load_anf_node == nullptr) {
      continue;
    }

    auto slice_anf_node = FindPrimCnode(func_graph, load_anf_node, prim::kPrimStridedSlice);
    if (slice_anf_node == nullptr) {
      continue;
    }

    auto status = this->RemoveSliceOp(manager, slice_anf_node, load_anf_node);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to remove slice op";
      return false;
    }

    for (auto &drop_node : this->remove_cnode_) {
      MS_LOG(INFO) << "Drop node :" << drop_node->fullname_with_scope();
      func_graph->DropNode(drop_node);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
