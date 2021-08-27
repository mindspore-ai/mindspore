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

#include <vector>
#include <memory>
#include "tools/optimizer/fisson/eliminate_concat_split.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/split_with_overlap.h"
#include "ops/concat.h"
#include "base/core_ops.h"
#include "tools/optimizer/parallel/spliter.h"

namespace mindspore {
namespace opt {
const BaseRef EliminateConcatSplit::DefinePattern() const {
  auto concat_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConcat>);
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();

  return VectorRef({split_prim, concat_var});
}

CNodePtr GetRealPrevCNode(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsRealCNodeKernel(cnode)) {
    return cnode;
  }

  auto input0 = cnode->input(0);
  if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
    auto temp_node = cnode->input(1);
    if (temp_node == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      return nullptr;
    }
    return GetRealPrevCNode(temp_node);
  } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
    return GetRealPrevCNode(cnode->input(1));
  } else {
    return nullptr;
  }
}

void ConcatSplitEliminate(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto pre_cnode = GetRealPrevCNode(cnode->input(1));
  if (pre_cnode == nullptr || !CheckPrimitiveType(pre_cnode, prim::kPrimConcat)) {
    return;
  }
  std::unordered_map<std::string, std::vector<AnfNodePtr>> graph_node_outputs =
    Spliter::GetInstance()->graph_node_outputs();
  auto finder = graph_node_outputs.find(pre_cnode->fullname_with_scope());
  if (finder == graph_node_outputs.end()) {
    return;
  }
  if (finder->second.size() > 1) return;

  size_t pre_inputs_size = pre_cnode->inputs().size();
  int pre_inputs_node_size = pre_inputs_size - 1;
  auto pre_prim = GetValueNode<std::shared_ptr<ops::Concat>>(pre_cnode->input(kAnfPrimitiveIndex));
  auto prim = GetValueNode<std::shared_ptr<ops::SplitWithOverlap>>(cnode->input(kAnfPrimitiveIndex));
  if (prim->get_number_split() != pre_inputs_node_size) {
    return;
  }

  // check axis NHWC
  // only support axis "N" now, other axes will support when having "InferShape"
  if (pre_prim->get_axis() != 0) {
    return;
  }

  // get inputs node
  auto it = graph_node_outputs.find(cnode->fullname_with_scope());
  if (it == graph_node_outputs.end()) {
    return;
  }
  int out_num = it->second.size();
  if (out_num != prim->get_number_split()) {
    return;
  }

  std::vector<CNodePtr> inputs_node;
  for (int i = 0; i < out_num; i++) {
    auto tmp = it->second[i];
    auto tmp_cnode = tmp->cast<CNodePtr>();
    if (tmp_cnode == nullptr) {
      return;
    }
    if (!CheckPrimitiveType(tmp_cnode, prim::kPrimTupleGetItem)) {
      return;
    }
    auto tmp_it = graph_node_outputs.find(tmp_cnode->fullname_with_scope());
    if (tmp_it == graph_node_outputs.end()) {
      return;
    }
    if (tmp_it->second.size() != 1) return;

    auto next = tmp_it->second[0];
    auto next_cnode = next->cast<CNodePtr>();

    inputs_node.push_back(next_cnode);
  }
  // replace inputs
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return;
  }
  for (size_t i = 1; i < pre_inputs_size; i++) {
    (void)manager->Replace((inputs_node[i - 1])->input(1), pre_cnode->input(i));
  }
}

const AnfNodePtr EliminateConcatSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_LOG(DEBUG) << "Enter EliminateConcatSplit pass process";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK) {
    return nullptr;
  }
  if (CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  auto split_cnode = node->cast<CNodePtr>();
  if (split_cnode == nullptr) {
    return nullptr;
  }
  ConcatSplitEliminate(func_graph, split_cnode);

  return node;
}
}  // namespace opt
}  // namespace mindspore
