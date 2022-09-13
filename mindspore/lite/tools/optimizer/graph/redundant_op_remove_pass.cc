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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include <memory>
#include <vector>
#include <utility>
#include "include/errorcode.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/make_tuple.h"
#include "ops/depend.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const size_t kIndexNum = 2;
int ProcessInputIsMonad(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto first_input = cnode->input(1);
  MS_ASSERT(first_input != nullptr);
  if (CheckPrimitiveType(first_input, prim::kPrimTranspose)) {
    first_input = cnode->input(1)->cast<CNodePtr>()->input(1);
    MS_CHECK_TRUE_MSG(first_input != nullptr, RET_ERROR, "first_input is nullptr");
  }
  auto second_input = cnode->input(kInputIndexTwo);
  MS_ASSERT(seconde_input != nullptr);
  if (CheckPrimitiveType(second_input, prim::kPrimTranspose)) {
    second_input = cnode->input(kInputIndexTwo)->cast<CNodePtr>()->input(1);
    MS_CHECK_TRUE_MSG(second_input != nullptr, RET_ERROR, "second_input is nullptr");
  }
  AnfNodePtr must_monad = nullptr;
  AnfNodePtr not_must_monad = nullptr;
  if (utils::isa<ValueNode>(first_input)) {
    auto value_node = first_input->cast<ValueNodePtr>();
    MS_ASSERT(value_node->value() != nullptr);
    if (utils::isa<Monad>(value_node->value())) {
      must_monad = first_input;
      not_must_monad = second_input;
    }
  }
  if (utils::isa<ValueNode>(second_input)) {
    auto value_node = second_input->cast<ValueNodePtr>();
    MS_ASSERT(value_node->value() != nullptr);
    if (utils::isa<Monad>(value_node->value())) {
      must_monad = second_input;
      not_must_monad = first_input;
    }
  }
  if (must_monad == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (!utils::isa<CNode>(not_must_monad) || CheckIsAllInputsParam(not_must_monad)) {
    manager->Replace(cnode, must_monad);
  } else {
    manager->Replace(cnode, not_must_monad);
  }
  return lite::RET_OK;
}

int ProcessDependencyWithTwoNodes(const FuncGraphPtr &func_graph, const CNodePtr &cnode, bool pre_node_is_first) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr pre_node = cnode->input(1);
  AnfNodePtr post_node = cnode->input(kInputIndexTwo);
  MS_ASSERT(pre_node != nullptr);
  MS_ASSERT(post_node != nullptr);
  if (!pre_node_is_first) {
    pre_node = cnode->input(kInputIndexTwo);
    post_node = cnode->input(1);
  }
  if (CheckPrimitiveType(pre_node, prim::kPrimTranspose)) {
    pre_node = cnode->input(1)->cast<CNodePtr>()->input(1);
    MS_CHECK_TRUE_MSG(pre_node != nullptr, RET_ERROR, "pre_node is nullptr");
  }
  if (CheckPrimitiveType(post_node, prim::kPrimTranspose)) {
    post_node = cnode->input(kInputIndexTwo)->cast<CNodePtr>()->input(1);
    MS_CHECK_TRUE_MSG(post_node != nullptr, RET_ERROR, "post_node is nullptr");
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_users = manager->node_users()[pre_node];
  auto iter =
    std::find_if(node_users.begin(), node_users.end(),
                 [&post_node](const std::pair<AnfNodePtr, int> &post_pair) { return post_pair.first == post_node; });
  if (iter == node_users.end()) {
    return lite::RET_NO_CHANGE;
  }
  auto tr = manager->Transact();
  tr.SetEdge(post_node, iter->second, NewValueNode(std::make_shared<UMonad>()));
  tr.Commit();
  auto depend_prim = std::make_shared<ops::Depend>();
  MS_CHECK_TRUE_MSG(depend_prim != nullptr, lite::RET_ERROR, "New Depend ops Failed");
  auto depend_prim_c = depend_prim->GetPrim();
  MS_CHECK_TRUE_MSG(depend_prim_c != nullptr, lite::RET_ERROR, "GetPrim Failed");
  auto depend_node = func_graph->NewCNode(depend_prim_c, {post_node, pre_node});
  MS_CHECK_TRUE_MSG(depend_node != nullptr, lite::RET_ERROR, "NewCNode Failed");
  depend_node->set_fullname_with_scope(cnode->fullname_with_scope());
  manager->Replace(cnode, depend_node);
  return lite::RET_OK;
}

int ProcessInputHaveDependency(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (ProcessDependencyWithTwoNodes(func_graph, cnode, true) == lite::RET_OK) {
    return lite::RET_OK;
  }
  if (ProcessDependencyWithTwoNodes(func_graph, cnode, false) == lite::RET_OK) {
    return lite::RET_OK;
  }
  auto make_tuple_node = std::make_shared<ops::MakeTuple>();
  MS_CHECK_TRUE_MSG(make_tuple_node != nullptr, lite::RET_ERROR, "make tuple node Failed");
  auto make_tuple_prim_c = make_tuple_node->GetPrim();
  MS_CHECK_TRUE_MSG(make_tuple_prim_c != nullptr, lite::RET_ERROR, "make tuple prim c Failed");
  auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
  MS_CHECK_TRUE_MSG(make_tuple_prim != nullptr, lite::RET_ERROR, "NewCNode Failed");
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (CheckPrimitiveType(cnode->input(0), prim::kPrimTranspose)) {
    manager->Replace(cnode->input(0)->cast<CNodePtr>()->input(0), make_tuple_prim);
    return RET_OK;
  }
  manager->Replace(cnode->input(0), make_tuple_prim);
  return lite::RET_OK;
}
}  // namespace

int RemoveRedundantOpPass::ReplaceOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  MS_CHECK_TRUE_MSG(anf_node != nullptr, RET_ERROR, "anf_node is nullptr");
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr");
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);
  if (CheckPrimitiveType(anf_node, kPrimIdentity)) {
    if (cnode->size() != kInputSizeTwo) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 1";
      remove_cnode_.insert(anf_node);
      return lite::RET_NO_CHANGE;
    }
  }
  if (CheckPrimitiveType(anf_node, prim::kPrimDepend)) {
    if (cnode->size() != kInputSizeTwo) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 1";
      remove_cnode_.insert(anf_node);
      return lite::RET_NO_CHANGE;
    }
  }
  if (CheckPrimitiveType(anf_node, prim::kPrimTranspose)) {
    if (cnode->size() != kInputSizeThree) {
      MS_LOG(DEBUG) << "The node inputs size is bigger than 2";
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
  MS_ASSERT(cnode != nullptr);
  if (ProcessInputIsMonad(func_graph, cnode) == lite::RET_OK) {
    return lite::RET_OK;
  }
  // both of two inputs are not monad, but have dependency.
  return ProcessInputHaveDependency(func_graph, cnode);
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
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().size() != kInputSizeThree) {
    MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << cnode->inputs().size();
    return RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode->input(1), kPrimIdentity)) {
    return lite::RET_NO_CHANGE;
  }
  auto get_item_input_cnode = cnode->input(1)->cast<CNodePtr>();
  auto index_vnode = cnode->input(kInputIndexTwo);
  if (!utils::isa<ValueNode>(index_vnode)) {
    MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(!CastToInt(index_vnode->cast<ValueNodePtr>()->value()).empty(), RET_ERROR, "value is empty");
  int index = CastToInt(index_vnode->cast<ValueNodePtr>()->value()).front();
  int input_cnode_inputs_size = static_cast<int>(get_item_input_cnode->inputs().size());
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

int RemoveRedundantOpPass::RemoveDropoutOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  MS_ASSERT(anf_node != nullptr);
  MS_ASSERT(manager != nullptr);
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() > kInputSizeTwo) {
    MS_LOG(ERROR) << "dropout input invalid.";
    return lite::RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTuplePtr>(anf_node->abstract())) {
    MS_LOG(DEBUG) << "dropout output size is one.";
    manager->Replace(anf_node, cnode->input(1));
  } else {
    auto node_users = manager->node_users()[anf_node];
    for (auto &node_user : node_users) {
      auto node = node_user.first;
      if (!CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
        MS_LOG(ERROR) << "dropout out node is invalid.";
        return lite::RET_ERROR;
      }
      auto get_index_node = node->cast<CNodePtr>()->input(kInputIndexTwo)->cast<ValueNodePtr>();
      if (get_index_node == nullptr) {
        MS_LOG(ERROR) << "tuple get item node is invalid.";
        return lite::RET_ERROR;
      }
      auto get_index = CastToInt(get_index_node->value()).front();
      if (get_index > 0 && !manager->node_users()[node].empty()) {
        MS_LOG(DEBUG) << "dropout's second output is useful.";
        continue;
      }
      manager->Replace(node, cnode->input(1));
    }
  }
  return lite::RET_OK;
}

int RemoveRedundantOpPass::GetConstDataFromInputNode(const CNodePtr &cnode, lite::DataInfo *data_info) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(data_info != nullptr);
  auto padding_node = cnode->input(kInputIndexTwo);
  MS_ASSERT(padding_node != nullptr);
  if (utils::isa<Parameter>(padding_node)) {
    auto status = lite::FetchDataFromParameterNode(cnode, kIndexNum, converter::kFmkTypeMs, data_info, true);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "fetch data from parameter node failed.";
      return lite::RET_ERROR;
    }
  } else if (utils::isa<ValueNode>(padding_node)) {
    auto status = lite::FetchDataFromValueNode(cnode, kIndexNum, converter::kFmkTypeMs, false, data_info, true);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "fetch data from value node failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int RemoveRedundantOpPass::RemoveInvalidPadOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(DEBUG) << "anf node is node a cnode.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);
  auto primitive = GetValueNode<mindspore::PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr:" << cnode->fullname_with_scope();
    return lite::RET_NO_CHANGE;
  }
  auto is_invalid = true;
  if (cnode->size() > kInputSizeTwo) {
    lite::DataInfo data_info;
    if (GetConstDataFromInputNode(cnode, &data_info) != RET_OK) {
      MS_LOG(ERROR) << "Get pad data failed.";
      return lite::RET_ERROR;
    }
    if (!data_info.data_.empty()) {
      auto pad_data = reinterpret_cast<int *>(data_info.data_.data());
      size_t num = data_info.data_.size() / sizeof(int);
      for (size_t i = 0; i < num; ++i) {
        if (pad_data[i] != 0) {
          is_invalid = false;
          break;
        }
      }
    } else {
      is_invalid = false;
    }
  } else {
    auto pad_prim = api::MakeShared<mindspore::ops::PadFusion>(primitive);
    MS_CHECK_TRUE_RET(pad_prim != nullptr, lite::RET_ERROR);
    MS_CHECK_TRUE_RET(pad_prim->GetAttr(ops::kPaddings) != nullptr, lite::RET_ERROR);
    auto pad_data = pad_prim->get_paddings();
    for (size_t i = 0; i < pad_data.size(); i++) {
      for (size_t j = 0; j < pad_data[i].size(); j++) {
        if (pad_data[i][j] != 0) {
          is_invalid = false;
          break;
        }
      }
      if (is_invalid == false) {
        break;
      }
    }
  }
  if (is_invalid) {
    return ReplaceOp(anf_node, manager);
  }
  return lite::RET_OK;
}

int RemoveRedundantOpPass::RemoveInvalidTransposeOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  auto cnode = anf_node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(DEBUG) << "The node inputs size is bigger than 2";
    return lite::RET_NO_CHANGE;
  }
  auto index_node = cnode->inputs()[kInputIndexTwo]->cast<ParameterPtr>();
  if (index_node == nullptr) {
    return RET_OK;
  }
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(index_node->default_param());
  MS_ASSERT(tensor_info != nullptr);
  if (tensor_info->Size() != 0) {
    return RET_OK;
  }
  return ReplaceOp(anf_node, manager);
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
    if (!is_train_model_ && CheckPrimitiveType(node, prim::kPrimDropout)) {
      status = RemoveDropoutOp(node, manager);
    }
    if (CheckPrimitiveType(node, prim::kPrimPadFusion)) {
      status = RemoveInvalidPadOp(node, manager);
    }
    if (CheckPrimitiveType(node, prim::kPrimTranspose)) {
      status = RemoveInvalidTransposeOp(node, manager);
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
