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
#include "tools/graph_kernel/converter/eliminate_redundant_op.h"
#include <algorithm>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::graphkernel {
namespace {
bool EliminateReshape(const CNodePtr &cnode, const FuncGraphManagerPtr &mng) {
  // Reshape + FastGeLU + Reshape --> FastGeLU
  auto input = cnode->input(kIndex1);
  if (!IsPrimitiveCNode(input, prim::kPrimReshape) || mng->node_users()[input].size() > 1) {
    return false;
  }
  auto users = mng->node_users()[cnode];
  if (users.size() != 1) {
    return false;
  }
  auto user = users.begin()->first;
  if (!IsPrimitiveCNode(user, prim::kPrimReshape)) {
    return false;
  }
  auto cb = Callback::Instance();
  auto input_in_shape = cb->GetInputShape(input, 0);
  auto user_out_shape = cb->GetOutputShape(user, 0);
  if (input_in_shape == user_out_shape) {
    MS_LOG(INFO) << "Eliminate Reshape around: " << cnode->fullname_with_scope();
    auto input_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    auto input_in_node = input_cnode->input(kIndex1);
    MS_EXCEPTION_IF_NULL(input_in_node);
    cnode->set_input(kIndex1, input_in_node);
    cnode->set_abstract(input_in_node->abstract()->Clone());
    (void)mng->Replace(user, cnode);
    return true;
  }
  return false;
}

bool EliminateTranspose(const CNodePtr &cnode, const FuncGraphManagerPtr &mng) {
  // Reshape + Transpose + Reshape --> Reshape
  auto input = cnode->input(kIndex1);
  if (!IsPrimitiveCNode(input, prim::kPrimReshape) || mng->node_users()[input].size() > 1) {
    return false;
  }
  auto users = mng->node_users()[cnode];
  if (users.size() != 1) {
    return false;
  }
  auto user = users.begin()->first;
  if (!IsPrimitiveCNode(user, prim::kPrimReshape)) {
    return false;
  }
  std::vector<int64_t> perm_list;
  if (cnode->input(kIndex2)->isa<Parameter>()) {
    auto perm_para = cnode->input(kIndex2)->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(perm_para);
    auto perm_tensor = perm_para->default_param()->cast<tensor::TensorPtr>();
    auto perm = static_cast<int32_t *>(perm_tensor->data_ptr()->data());
    std::transform(perm, perm + perm_tensor->shape()[0], std::back_inserter(perm_list), IntToLong);
  } else {
    auto perm_value = cnode->input(kIndex2)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(perm_value);
    perm_list = GetValue<std::vector<int64_t>>(perm_value->value());
  }
  std::vector<int64_t> opt_perm = {0, 2, 1, 3};
  auto cb = Callback::Instance();
  auto x_shape = cb->GetInputShape(cnode, 0);
  if (perm_list == opt_perm && x_shape.size() == opt_perm.size() && (x_shape[kIndex1] == 1 || x_shape[kIndex2] == 1)) {
    MS_LOG(INFO) << "Eliminate Transpose: " << cnode->fullname_with_scope();
    auto user_cnode = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    auto input_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    user_cnode->set_input(kIndex1, input_cnode->input(kIndex1));
    return true;
  }
  return false;
}
}  // namespace

bool EliminateRedundantOp::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &node : todos) {
    if (node == nullptr) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimFastGeLU)) {
      changed = EliminateReshape(cnode, mng) || changed;
    } else if (IsPrimitiveCNode(cnode, prim::kPrimTranspose)) {
      changed = EliminateTranspose(cnode, mng) || changed;
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
