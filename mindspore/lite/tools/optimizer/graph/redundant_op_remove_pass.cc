/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
namespace {
constexpr size_t InputDoubleNum = 2;
constexpr size_t InputTripleNum = 3;
constexpr auto kNameLoad = "Load";
constexpr auto kNameUpdateState = "UpdateState";
}  // namespace
int RemoveRedundantOpPass::ReplaceOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto type = opt::GetCNodeType(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  if (type == schema::PrimitiveType_Identity) {
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

int RemoveRedundantOpPass::ReplaceTupleGetItem(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto type = opt::GetCNodeType(anf_node);
  if (type != schema::PrimitiveType_TupleGetItem) {
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode->inputs().size() != InputTripleNum) {
    MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << cnode->inputs().size();
    return RET_ERROR;
  }
  type = opt::GetCNodeType(cnode->input(1));
  if (type != schema::PrimitiveType_Identity) {
    return lite::RET_NO_CHANGE;
  }
  auto get_item_input_cnode = cnode->input(1)->cast<CNodePtr>();
  auto index_vnode = cnode->input(2);
  if (!utils::isa<ValueNode>(index_vnode)) {
    MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
    return lite::RET_ERROR;
  }
  int index = lite::CastToInt(index_vnode->cast<ValueNodePtr>()->value()).front();
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
    auto type = opt::GetCNodeType(node);
    if (type == schema::PrimitiveType_Identity) {
      status = ReplaceOp(node, manager);
    }
    if (CheckPrimitiveType(node, std::make_shared<Primitive>(kNameLoad))) {
      status = ReplaceOp(node, manager);
    }
    if (CheckPrimitiveType(node, std::make_shared<Primitive>(kNameUpdateState))) {
      status = ReplaceOp(node, manager);
    }
    if (type == schema::PrimitiveType_Depend ||
        type == schema::PrimitiveType_ControlDepend) {  // ControlDepend delete next version.
      status = ReplaceOp(node, manager);
    }
    if (type == schema::PrimitiveType_TupleGetItem) {
      status = ReplaceTupleGetItem(node, manager);
    }
    if (type == schema::PrimitiveType_If || type == schema::PrimitiveType_While) {
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
