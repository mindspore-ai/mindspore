/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/multi_matmuls_fusion.h"

#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
namespace opt {
bool MultiMatmulsFusion::Run(const FuncGraphPtr &graph) {
  bool changed = false;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost() || common::GetEnv("ENABLE_MATMUL_FUSION") != "on") {
    return changed;
  }

  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  const auto &node_users_map = mng->node_users();
  auto node_list = TopoSort(graph->output());

  for (const auto &node : node_list) {
    AnfNodePtrList user_matmuls;
    if (node_users_map.find(node) == node_users_map.end()) continue;
    for (const auto &user_pair : node_users_map.at(node)) {
      // the node is MatMul's first input.
      if (IsPrimitiveCNode(user_pair.first, prim::kPrimMatMul) && user_pair.second == 1) {
        user_matmuls.push_back(user_pair.first);
      }
    }
    if (user_matmuls.size() <= 1) {
      continue;
    }
    AnfNodePtrList getitems;
    if (user_matmuls.size() == 2) {
      Process("MatmulFfn", node, user_matmuls, &getitems);
    } else if (user_matmuls.size() == 3) {
      Process("MatmulQkv", node, user_matmuls, &getitems);
    } else {
      MS_LOG(INFO) << "user_matmuls.size() == " << user_matmuls.size();
    }
    if (!getitems.empty()) {
      for (size_t i = 0; i < getitems.size(); i++) {
        (void)mng->Replace(user_matmuls[i], getitems[i]);
      }
      changed = true;
    }
  }
  return changed;
}

void MultiMatmulsFusion::Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
                                 AnfNodePtrList *getitems) const {
  AnfNodePtrList fused_inputs = {NewValueNode(std::make_shared<Primitive>(name)), node};
  abstract::AbstractBasePtrList new_abs;
  for (auto &user : users) {
    auto matmul = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul);
    // insert "B, trans_a, trans_b"
    fused_inputs.insert(fused_inputs.end(), matmul->inputs().begin() + 2, matmul->inputs().end());
    new_abs.push_back(user->abstract());
  }
  auto fused_matmul = node->func_graph()->NewCNode(fused_inputs);
  fused_matmul->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abs));
  for (size_t i = 0; i < users.size(); i++) {
    // create getitem(i)
    auto idx_val = MakeValue(SizeToLong(i));
    auto idx = NewValueNode(idx_val);
    idx->set_abstract(idx_val->ToAbstract());
    auto getitem = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fused_matmul, idx});
    (void)getitems->emplace_back(getitem);
  }
}
}  // namespace opt
}  // namespace mindspore
