/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tf/functionalize_cond.h"
#include <algorithm>
#include <memory>
#include <deque>
#include <unordered_set>
#include "include/errorcode.h"
#include "tools/converter/ops/ops_def.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/return.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore::opt {
STATUS FunctionalizeCond::GetSwitchBranchType(const CNodePtr &switch_cnode, BranchType *branch_type) {
  MS_ASSERT(switch_cnode != nullptr);
  MS_ASSERT(branch_type != nullptr);
  auto manager = fg_->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto node_users = manager->node_users()[switch_cnode];
  if (node_users.size() != 1) {  // only one output of switch is referenced in cond
    MS_LOG(ERROR) << "switch's node users is not correct";
    return RET_ERROR;
  }
  auto node_user = node_users.front();
  auto tuple_get_item = node_user.first;
  if (!utils::isa<CNodePtr>(tuple_get_item) || !CheckPrimitiveType(tuple_get_item, prim::kPrimTupleGetItem)) {
    MS_LOG(ERROR) << "switch's node user is not TupleGetItem";
    return RET_ERROR;
  }
  auto tuple_get_item_cnode = utils::cast<CNodePtr>(tuple_get_item);
  auto idx = GetTupleGetItemOutIndex(tuple_get_item_cnode);
  if (idx == 0) {
    *branch_type = kElseBranch;
  } else if (idx == 1) {
    *branch_type = kThenBranch;
  } else {
    MS_LOG(ERROR) << "wrong tuple_get_item index";
    return RET_ERROR;
  }
  return RET_OK;
}

void FunctionalizeCond::CheckBranchIsEffective(const CNodePtr &switch_cnode, BranchType branch_type) {
  MS_ASSERT(switch_cnode != nullptr);
  MS_ASSERT(is_effective != nullptr);
  if (switch_cnode->size() < C3NUM) {
    return;
  }
  auto cond_node = switch_cnode->input(kInputIndexTwo);
  if (!utils::isa<Parameter>(cond_node)) {
    return;
  }
  auto cond_pnode = cond_node->cast<ParameterPtr>();
  lite::DataInfo data_info;
  if (FetchFromDefaultParam(cond_pnode, converter::FmkType::kFmkTypeTf, &data_info, false) != lite::RET_OK) {
    return;
  }
  if (data_info.data_ptr_ == nullptr || data_info.data_type_ != kNumberTypeBool) {
    return;
  }
  bool cond = *(static_cast<bool *>(data_info.data_ptr_));
  if (branch_type == kThenBranch) {
    then_switch_ = switch_cnode;
    then_is_effective_ = cond;
  }
  if (branch_type == kElseBranch) {
    else_switch_ = switch_cnode;
    else_is_effective_ = !cond;
  }
}

STATUS FunctionalizeCond::BranchSubGraphAddNodes(const FuncGraphPtr &graph, const AnfNodePtr &root_node,
                                                 BranchType branch_type) {
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(root_node);
  std::deque<AnfNodePtr> q;
  std::unordered_set<AnfNodePtr> vis;
  q.push_back(root_node);
  while (!q.empty()) {
    auto node = q.front();
    CHECK_NULL_RETURN(node);
    q.pop_front();
    vis.insert(node);
    if (FunctionalizeControlOpPass::IsSwitch(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      BranchType this_type;
      if (GetSwitchBranchType(cnode, &this_type) != RET_OK || this_type != branch_type) {
        MS_LOG(ERROR) << "switch node in branch " << branch_type << " is not correct";
        return RET_ERROR;
      }
      CheckBranchIsEffective(cnode, branch_type);
      continue;
    }
    if (utils::isa<ParameterPtr>(node)) {
      graph->add_parameter(node->cast<ParameterPtr>());
    }
    graph->AddNode(node);
    if (!utils::isa<ValueNodePtr>(node)) {
      node->set_func_graph(graph);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto inputi = cnode->input(i);
        if (vis.find(inputi) == vis.end()) {
          q.push_back(cnode->input(i));
        }
      }
    }
  }
  return RET_OK;
}

int FunctionalizeCond::PosInInputNodes(const CNodePtr &node) {
  auto index = std::find(input_nodes_.begin(), input_nodes_.end(), node);
  if (index == input_nodes_.end()) {
    input_nodes_.push_back(node);
    return input_nodes_.size() - 1;
  }
  return index - input_nodes_.begin();
}

STATUS FunctionalizeCond::IdentifySubgraphInput(const FuncGraphPtr &graph, std::string graph_name) {
  CHECK_NULL_RETURN(graph);
  std::vector<AnfNodePtr> nodes_need_drop{};
  for (auto &cnode : graph->GetOrderedCnodes()) {
    for (auto &input_node : cnode->inputs()) {
      if (FunctionalizeControlOpPass::IsSwitch(input_node)) {
        CHECK_NULL_RETURN(input_node);
        auto switch_node = input_node->cast<CNodePtr>();
        CHECK_NULL_RETURN(switch_node);
        auto switch_input = utils::cast<CNodePtr>(switch_node->input(1));
        auto pos = PosInInputNodes(switch_input);
        nodes_need_drop.push_back(cnode);
        pred_nodes_.push_back(switch_node->input(kInputIndexTwo));
        // set parameter
        auto parameter = graph->add_parameter();
        CHECK_NULL_RETURN(parameter);
        parameter->set_abstract(cnode->abstract());
        // hardcode for subgraph input name
        parameter->set_name(graph_name + "_input_" + std::to_string(pos) + "_parameter");

        // replace switch
        auto manager = fg_->manager();
        CHECK_NULL_RETURN(manager);
        auto node_users = manager->node_users()[cnode];
        for (auto &node_user : node_users) {
          if (graph->nodes().contains(node_user.first)) {
            manager->SetEdge(node_user.first, node_user.second, parameter);
          }
        }
      }
    }
  }
  return RET_OK;
}

FuncGraphPtr FunctionalizeCond::CreateBranchGraph(const AnfNodePtr &node, std::string name, BranchType branch_type) {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto graph = FunctionalizeControlOpPass::NewFuncGraph(name, converter::kFmkTypeTf);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "new graph Partial Node return nullptr";
    return nullptr;
  }
  graph->set_manager(fg_->manager());
  auto status = BranchSubGraphAddNodes(graph, node, branch_type);
  if (status != RET_OK) {
    return nullptr;
  }

  if (!CheckPrimitiveType(node, prim::kPrimSwitch)) {  // graph is not empty
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return nullptr;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    MS_CHECK_TRUE_RET(return_prim_c != nullptr, nullptr);
    auto value_node = NewValueNode(return_prim_c);
    MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
    std::vector<AnfNodePtr> op_inputs{value_node, node};  // If subgraph only has one output tensor
    auto return_cnode = graph->NewCNode(op_inputs);
    MS_CHECK_TRUE_RET(return_cnode != nullptr, nullptr);
    return_cnode->set_fullname_with_scope(name + "-return");
    return_cnode->set_func_graph(graph);
    graph->set_return(return_cnode);
    auto graph_output = graph->output();
    MS_CHECK_TRUE_RET(graph_output != nullptr, nullptr);
    auto graph_output_cnode = graph_output->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(graph_output_cnode != nullptr, nullptr);
    graph_output_cnode->set_fullname_with_scope(name + "_output_0_cnode");
  }
  return graph;
}

CNodePtr FunctionalizeCond::CreateNewIf(const FuncGraphPtr &else_branch, const FuncGraphPtr &then_branch) {
  MS_CHECK_TRUE_RET(else_branch != nullptr, nullptr);
  MS_CHECK_TRUE_RET(then_branch != nullptr, nullptr);

  auto if_primc = std::make_shared<mindspore::lite::If>();
  if (if_primc == nullptr) {
    MS_LOG(ERROR) << "new if_primitive failed";
    return nullptr;
  }
  auto if_value_node = NewValueNode(if_primc);
  if (if_value_node == nullptr) {
    MS_LOG(ERROR) << "new if_value_node failed";
    return nullptr;
  }
  auto then_value_node = NewValueNode(then_branch);
  MS_CHECK_TRUE_RET(then_value_node != nullptr, nullptr);
  auto else_value_node = NewValueNode(else_branch);
  MS_CHECK_TRUE_RET(else_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> if_op_inputs = {if_value_node, then_value_node, else_value_node, pred_node_};
  std::copy(input_nodes_.begin(), input_nodes_.end(), std::back_inserter(if_op_inputs));
  return fg_->NewCNode(if_op_inputs);
}

STATUS FunctionalizeCond::VerifyPredictNode() {
  if (pred_nodes_.empty()) {
    return RET_ERROR;
  }
  for (size_t i = 1; i < pred_nodes_.size(); ++i) {
    if (pred_nodes_[i] != pred_nodes_[0]) {
      return RET_ERROR;
    }
  }
  pred_node_ = pred_nodes_[0];
  return RET_OK;
}

STATUS FunctionalizeCond::DegenerateNonControlFlow(const FuncGraphPtr &else_graph, const FuncGraphPtr &then_graph) {
  MS_ASSERT(else_graph != nullptr && then_graph != nullptr);
  std::vector<AnfNodePtr> nodes;
  auto else_nodes = else_graph->nodes();
  nodes.insert(nodes.end(), else_nodes.begin(), else_nodes.end());
  auto then_nodes = then_graph->nodes();
  nodes.insert(nodes.end(), then_nodes.begin(), then_nodes.end());
  for (auto &node : nodes) {
    MS_CHECK_TRUE_MSG(node != nullptr, lite::RET_ERROR, "find a node is a nullptr.");
    if (!utils::isa<ValueNode>(node)) {
      node->set_func_graph(fg_);
    }
  }
  auto manager = fg_->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "manager must be not a nullptr.");
  CNodePtr switch_op{nullptr};
  int merge_input_index = 1;
  if (then_is_effective_ && !else_is_effective_) {
    switch_op = then_switch_;
    merge_input_index = kInputIndexTwo;
  } else if (else_is_effective_ && !then_is_effective_) {
    switch_op = else_switch_;
  } else {
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(switch_op != nullptr, lite::RET_NULL_PTR, "switch node is a nullptr.");
  MS_CHECK_TRUE_MSG(switch_op->size() >= kInputSizeThree, lite::RET_ERROR, "switch's inputs-size is invalid.");
  auto node_users = manager->node_users()[switch_op];
  for (auto &node_user : node_users) {
    auto post_node = node_user.first;
    if (!CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
      MS_LOG(ERROR) << "switch's post-node must be TupleGetItem.";
      return lite::RET_ERROR;
    }
    if (!manager->Replace(post_node, switch_op->input(1))) {
      MS_LOG(ERROR) << "Manager: Replace unused switch-node failed.";
      return lite::RET_ERROR;
    }
  }
  if (!manager->Replace(merge_node_, merge_node_->input(merge_input_index))) {
    MS_LOG(ERROR) << "Manager: Replace unused merge-node failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS FunctionalizeCond::Process() {
  if (fg_ == nullptr || merge_node_ == nullptr || merge_node_->inputs().size() != kInputSizeThree) {
    MS_LOG(ERROR) << "fg or merge is not correct";
    return RET_ERROR;
  }

  then_is_effective_ = true;
  then_switch_ = nullptr;
  else_is_effective_ = true;
  else_switch_ = nullptr;
  auto else_branch_name = merge_node_->fullname_with_scope() + "-partial-if-else";
  auto then_branch_name = merge_node_->fullname_with_scope() + "-partial-then-else";

  auto else_branch = CreateBranchGraph(merge_node_->input(1), else_branch_name, kElseBranch);
  if (else_branch == nullptr) {
    MS_LOG(ERROR) << "create else branch failed";
    return RET_ERROR;
  }
  auto then_branch = CreateBranchGraph(merge_node_->input(kInputIndexTwo), then_branch_name, kThenBranch);
  if (then_branch == nullptr) {
    MS_LOG(ERROR) << "create then branch failed";
    return RET_ERROR;
  }
  if (else_is_effective_ ^ then_is_effective_) {
    auto status = DegenerateNonControlFlow(else_branch, then_branch);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "Degenerate to non-control-flow failed.";
    }
    return status;
  }

  auto status = IdentifySubgraphInput(else_branch, else_branch_name);
  if (status != RET_OK) {
    return status;
  }
  status = IdentifySubgraphInput(then_branch, then_branch_name);
  if (status != RET_OK) {
    return status;
  }

  status = VerifyPredictNode();
  if (status != RET_OK) {
    return status;
  }

  auto if_node = CreateNewIf(else_branch, then_branch);
  if (if_node == nullptr) {
    MS_LOG(ERROR) << "create if node error";
    return RET_ERROR;
  }
  if_node->set_abstract(merge_node_->abstract()->Clone());
  auto manager = fg_->manager();
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[merge_node_];
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, if_node);
  }
  return RET_OK;
}
}  // namespace mindspore::opt
