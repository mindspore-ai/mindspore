/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include <memory>
#include <vector>
#include "include/errorcode.h"
#include "ops/make_tuple.h"

namespace mindspore::opt {
namespace {
constexpr size_t InputDoubleNum = 2;
constexpr size_t InputTripleNum = 3;
void FetchCNodeFromMakeTuple(const AnfNodePtr &anf_node, std::vector<AnfNodePtr> *inputs) {
  MS_ASSERT(anf_node != nullptr);
  MS_ASSERT(inputs != nullptr);
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (cnode->input(i)->isa<CNode>()) {
      inputs->push_back(cnode->input(i));
    }
  }
}
}  // namespace
int RemoveRedundantOpPass::ReplaceOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (CheckPrimitiveType(anf_node, kPrimIdentity)) {
    if (cnode->size() != InputDoubleNum) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 1";
      remove_cnode_.insert(anf_node);
      return lite::RET_NO_CHANGE;
    }
  }
  if (CheckPrimitiveType(anf_node, prim::kPrimDepend)) {
    if (cnode->size() != InputDoubleNum) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 1";
      remove_cnode_.insert(anf_node);
      return lite::RET_NO_CHANGE;
    }
  }
  if (CheckPrimitiveType(anf_node, prim::kPrimControlDepend)) {
    if (cnode->size() != InputDoubleNum) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 1";
      remove_cnode_.insert(anf_node);
      return lite::RET_NO_CHANGE;
    }
  }

  bool replace_succ = manager->Replace(anf_node, cnode->input(1));
  if (!replace_succ) {
    MS_LOG(ERROR) << "replace redundant op failed.";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

int RemoveRedundantOpPass::ReplaceUpdateStateOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  auto inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!inputs[i]->isa<CNode>()) {
      continue;
    }
    if (CheckPrimitiveType(inputs[i], prim::kPrimMakeTuple)) {
      FetchCNodeFromMakeTuple(inputs[i], &new_inputs);
      continue;
    }
    new_inputs.push_back(inputs[i]);
  }
  for (auto &node : new_inputs) {
    func_graph->get_return()->add_input(node);
  }
  auto value = std::make_shared<UMonad>();
  bool replace_succ = func_graph->manager()->Replace(anf_node, NewValueNode(value));
  if (!replace_succ) {
    MS_LOG(ERROR) << "replace redundant op failed.";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

int RemoveRedundantOpPass::ReplaceTupleGetItem(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  if (!CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode->inputs().size() != InputTripleNum) {
    MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << cnode->inputs().size();
    return RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode->input(1), kPrimIdentity)) {
    return lite::RET_NO_CHANGE;
  }
  auto get_item_input_cnode = cnode->input(1)->cast<CNodePtr>();
  auto index_vnode = cnode->input(2);
  if (!utils::isa<ValueNode>(index_vnode)) {
    MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
    return lite::RET_ERROR;
  }
  int index = CastToInt(index_vnode->cast<ValueNodePtr>()->value()).front();
  int input_cnode_inputs_size = get_item_input_cnode->inputs().size();
  if ((index + 1) >= input_cnode_inputs_size) {
    MS_LOG(ERROR) << "value node index is out of range.";
    return lite::RET_ERROR;
  }
  bool replace_succ = manager->Replace(anf_node, get_item_input_cnode->input(index + 1));
  if (!replace_succ) {
    MS_LOG(ERROR) << "replace identity failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool RemoveRedundantOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, kPrimIdentity)) {
      status = ReplaceOp(node, manager);
    }
    if (CheckPrimitiveType(node, prim::kPrimLoad)) {
      status = ReplaceOp(node, manager);
    }
    if (CheckPrimitiveType(node, prim::kPrimUpdateState)) {
      status = ReplaceUpdateStateOp(func_graph, node);
    }
    if (CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      status = ReplaceTupleGetItem(node, manager);
    }
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(node->cast<CNodePtr>()->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      (void)Run(sub_func_graph);
      sub_func_graph = GetValueNode<FuncGraphPtr>(node->cast<CNodePtr>()->input(2));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      (void)Run(sub_func_graph);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "remove identity pass is failed.";
      return false;
    }
  }
  for (auto &node : remove_cnode_) {
    func_graph->DropNode(node);
  }
  return true;
}
}  // namespace mindspore::opt
