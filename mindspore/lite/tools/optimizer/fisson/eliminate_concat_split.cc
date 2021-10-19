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
#include <string>
#include <unordered_map>
#include "tools/optimizer/fisson/eliminate_concat_split.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/split_with_overlap.h"
#include "ops/concat.h"
#include "base/core_ops.h"
#include "tools/optimizer/parallel/spliter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr GetRealPrevCNode(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr && node->isa<CNode>());
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

int ConcatSplitEliminate(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto pre_cnode = GetRealPrevCNode(cnode->input(1));
  if (pre_cnode == nullptr || !CheckPrimitiveType(pre_cnode, prim::kPrimConcat)) {
    return RET_OK;
  }
  std::unordered_map<std::string, std::vector<AnfNodePtr>> graph_node_outputs =
    Spliter::GetInstance()->graph_node_outputs();
  auto finder = graph_node_outputs.find(pre_cnode->fullname_with_scope());
  if (finder == graph_node_outputs.end()) {
    return RET_ERROR;
  }
  if (finder->second.size() > 1) {
    return RET_OK;
  }

  size_t pre_inputs_size = pre_cnode->inputs().size();
  int pre_inputs_node_size = pre_inputs_size - 1;
  auto pre_prim = GetValueNode<std::shared_ptr<ops::Concat>>(pre_cnode->input(kAnfPrimitiveIndex));
  MS_CHECK_TRUE_MSG(pre_prim != nullptr, lite::RET_ERROR, "pre_cnode is not a ops::Concat");
  auto prim = GetValueNode<std::shared_ptr<ops::SplitWithOverlap>>(cnode->input(kAnfPrimitiveIndex));
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "cnode is not a ops::SplitWithOverlap");
  if (prim->get_number_split() != pre_inputs_node_size) {
    return RET_OK;
  }

  // check axis NHWC
  // only support axis "N" now, other axes will support when having "InferShape"
  if (pre_prim->get_axis() != 0) {
    return RET_OK;
  }

  // get inputs node
  auto it = graph_node_outputs.find(cnode->fullname_with_scope());
  if (it == graph_node_outputs.end()) {
    return RET_ERROR;
  }
  int out_num = it->second.size();
  if (out_num != prim->get_number_split()) {
    return RET_OK;
  }

  std::vector<CNodePtr> inputs_node;
  for (int i = 0; i < out_num; i++) {
    auto tmp = it->second[i];
    auto tmp_cnode = tmp->cast<CNodePtr>();
    if (tmp_cnode == nullptr) {
      return RET_ERROR;
    }
    if (!CheckPrimitiveType(tmp_cnode, prim::kPrimTupleGetItem)) {
      return RET_OK;
    }
    auto tmp_it = graph_node_outputs.find(tmp_cnode->fullname_with_scope());
    if (tmp_it == graph_node_outputs.end()) {
      return RET_ERROR;
    }
    if (tmp_it->second.size() != 1) {
      return RET_OK;
    }

    auto next = tmp_it->second[0];
    auto next_cnode = next->cast<CNodePtr>();
    MS_ASSERT(next_cnode != nullptr);
    inputs_node.push_back(next_cnode);
  }
  // replace inputs
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return RET_OK;
  }
  for (size_t i = 1; i < pre_inputs_size; i++) {
    if (!manager->Replace((inputs_node[i - 1])->input(1), pre_cnode->input(i))) {
      MS_LOG(DEBUG) << "Replace failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace

const BaseRef EliminateConcatSplit::DefinePattern() const {
  auto concat_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConcat>);
  CHECK_NULL_RETURN(concat_var);
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();
  CHECK_NULL_RETURN(split_prim);
  return VectorRef({split_prim, concat_var});
}

const AnfNodePtr EliminateConcatSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_LOG(DEBUG) << "Enter EliminateConcatSplit pass process";
  if (func_graph == nullptr) {
    return nullptr;
  }
  if (node == nullptr) {
    return nullptr;
  }
  auto split_cnode = node->cast<CNodePtr>();
  if (split_cnode == nullptr) {
    return nullptr;
  }
  auto ret = ConcatSplitEliminate(func_graph, split_cnode);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "ConcatSplitEliminate failed: " << ret;
    return nullptr;
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
