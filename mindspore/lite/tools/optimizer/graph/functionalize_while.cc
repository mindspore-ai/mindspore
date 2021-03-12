/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <memory>
#include <deque>
#include "tools/optimizer/graph/functionalize_while.h"
#include "include/errorcode.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"
#include "tools/converter/ops/while.h"

namespace {
mindspore::ValueNodePtr GetWhileAnfPrim() {
  auto while_primc = std::make_shared<mindspore::lite::While>();
  if (while_primc == nullptr) {
    MS_LOG(ERROR) << "new while_primitive failed";
    return nullptr;
  }
  while_primc->set_cond_subgraph_index(mindspore::opt::FunctionalizeControlOpPass::GetSubgraphIndex());
  while_primc->set_body_subgraph_index(mindspore::opt::FunctionalizeControlOpPass::GetSubgraphIndex());
  mindspore::ValueNodePtr partial_anf_prim = NewValueNode(while_primc);
  return partial_anf_prim;
}
}  // namespace

namespace mindspore::opt {

using mindspore::lite::RET_NULL_PTR;

CNodePtr FunctionalizeWhile::BlongToWhichSwitch(const CNodePtr &node) {
  return FunctionalizeControlOpPass::BelongToWhichNode(node, FunctionalizeControlOpPass::IsSwitch);
}
CNodePtr FunctionalizeWhile::BlongToWhichMerge(const CNodePtr &node) {
  return FunctionalizeControlOpPass::BelongToWhichNode(node, FunctionalizeControlOpPass::IsMerge);
}
CNodePtr FunctionalizeWhile::BlongToWhichEnter(const CNodePtr &node) {
  return FunctionalizeControlOpPass::BelongToWhichNode(node, FunctionalizeControlOpPass::IsEnter);
}

CNodePtr FunctionalizeWhile::BlongToWhichExternalEnter(const CNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  if (FunctionalizeControlOpPass::IsEnter(node)) {
    return node;
  }
  CNodePtr aim_node = nullptr;
  std::deque<AnfNodePtr> todo(256);
  todo.clear();
  for (auto &input_node : node->inputs()) {
    if (FunctionalizeControlOpPass::IsEnter(input_node) && WhileNodeExternalInputIsContain(input_node)) {
      aim_node = utils::cast<CNodePtr>(input_node);
      todo.clear();
      break;
    }
    todo.push_back(input_node);
  }

  while (!todo.empty()) {
    AnfNodePtr todo_node = todo.front();
    todo.pop_front();
    if (FunctionalizeControlOpPass::IsEnter(todo_node) && WhileNodeExternalInputIsContain(todo_node)) {
      aim_node = utils::cast<CNodePtr>(todo_node);
      todo.clear();
      break;
    }
    if (utils::isa<CNodePtr>(todo_node)) {
      auto cnode = utils::cast<CNodePtr>(todo_node);
      for (size_t i = 0; i < cnode->inputs().size(); i++) {
        todo.push_back(cnode->input(i));
      }
    }
  }
  if (aim_node == nullptr) {
    MS_LOG(WARNING) << "not found belonging enter node.";
    return nullptr;
  }

  return aim_node;
}

int FunctionalizeWhile::PosInInputEnterNodes(const CNodePtr &node) {
  auto index = std::find(input_enter_nodes_.begin(), input_enter_nodes_.end(), node);
  if (index == input_enter_nodes_.end()) {
    return POS_INVALID;
  }
  return index - input_enter_nodes_.begin();
}

STATUS FunctionalizeWhile::NewWhileNode() {
  ValueNodePtr while_anf_primitive = GetWhileAnfPrim();
  if (while_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "Get while anf primitive failed.";
    return RET_NULL_PTR;
  }

  static int count = 0;
  std::vector<AnfNodePtr> while_op_inputs = {while_anf_primitive};
  while_node_ = fg_->NewCNode(while_op_inputs);
  while_node_->set_fullname_with_scope(loop_cond_node_->fullname_with_scope() + "-while-" + std::to_string(count++));
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyWhileNodeInput() {
  for (auto &node : node_cluster_) {
    if (FunctionalizeControlOpPass::IsEnter(node)) {
      auto enter_cnode = node->cast<CNodePtr>();
      input_enter_nodes_.push_back(enter_cnode);
      while_node_->add_input(enter_cnode->input(1));
    }
  }
  if (input_enter_nodes_.empty()) {
    MS_LOG(ERROR) << "not found input of while node.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyWhileNodeExternalInput() {
  std::deque<AnfNodePtr> todo(128);
  std::vector<CNodePtr> merge_nodes{};
  todo.clear();
  for (size_t i = 1; i < loop_cond_node_->inputs().size(); i++) {
    todo.push_back(loop_cond_node_->input(i));
  }
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();
    if (FunctionalizeControlOpPass::IsMerge(node)) {
      merge_nodes.push_back(node->cast<CNodePtr>());
      continue;
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        todo.push_back(cnode->input(i));
      }
    }
  }

  for (auto &node : merge_nodes) {
    external_input_enter_nodes_.push_back(node->input(1)->cast<CNodePtr>());
  }
  return RET_OK;
}

bool FunctionalizeWhile::WhileNodeExternalInputIsContain(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  return std::find(external_input_enter_nodes_.begin(), external_input_enter_nodes_.end(), cnode) !=
         external_input_enter_nodes_.end();
}

STATUS FunctionalizeWhile::IdentifyWhileNodeOutput() {
  output_exit_nodes_.resize(external_input_enter_nodes_.size());
  for (auto &node : node_cluster_) {
    // exit ->switch->merge->enter
    if (FunctionalizeControlOpPass::IsExit(node)) {
      auto exit_node = node->cast<CNodePtr>();
      auto switch_node = BlongToWhichSwitch(exit_node);
      auto merge_node = BlongToWhichMerge(switch_node);
      auto enter_node = BlongToWhichExternalEnter(merge_node);
      int pos = PosInInputEnterNodes(enter_node);
      if (pos == -1) {
        MS_LOG(ERROR) << "not find in input enter nodes.";
        return RET_ERROR;
      }
      output_exit_nodes_.at(pos) = exit_node;
    }
  }

  if (output_exit_nodes_.size() == 1) {
    while_node_->set_abstract(output_exit_nodes_[0]->abstract());
  } else {
    AbstractBasePtrList abstract_list;
    abstract_list.resize(output_exit_nodes_.size());
    std::transform(output_exit_nodes_.begin(), output_exit_nodes_.end(), abstract_list.begin(),
                   [](const CNodePtr &cnode) { return cnode->abstract(); });
    while_node_->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::UpdateExitNodeUser() {
  auto manager = fg_->manager();
  if (output_exit_nodes_.size() == 1) {
    if (!manager->Replace(output_exit_nodes_[0], while_node_)) {
      MS_LOG(ERROR) << "replace node failed.";
      return RET_ERROR;
    }
  } else {
    for (auto &node : output_exit_nodes_) {
      auto node_users = manager->node_users()[node];
      for (auto &node_user : node_users) {
        // new getitem
        AbstractBasePtrList abstractList;
        std::vector<int64_t> shape_vector;
        abstractList.emplace_back(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector));
        auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
        if (tuple_get_item_prim_ptr == nullptr) {
          MS_LOG(ERROR) << "GetTupleGetItemPrim return nullptr";
          return RET_NULL_PTR;
        }
        auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
        const auto &exit_node = node;
        auto switch_node = BlongToWhichSwitch(exit_node);
        auto merge_node = BlongToWhichMerge(switch_node);
        auto enter_node = BlongToWhichExternalEnter(merge_node);
        int output_idx = PosInInputEnterNodes(enter_node);
        auto getItemValue = NewValueNode(MakeValue<int>(output_idx));
        std::vector<AnfNodePtr> inputs{tuple_get_item_prim, while_node_, getItemValue};
        CNodePtr get_item_node = fg_->NewCNode(inputs);
        std::string output_item_name = while_node_->fullname_with_scope() + "_getitem_" + std::to_string(output_idx);
        auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector);
        if (abstract == nullptr) {
          MS_LOG(ERROR) << "create AbstractTensor failed";
          return RET_NULL_PTR;
        }
        get_item_node->set_abstract(abstract);
        get_item_node->set_fullname_with_scope(output_item_name);
        // set
        if (fg_->nodes().contains(node_user.first)) {
          if (!manager->Replace(output_exit_nodes_[0], while_node_)) {
            MS_LOG(ERROR) << "replace node failed.";
            return RET_ERROR;
          }
        }
      }
    }
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::BuildWhileNode() {
  int ret = NewWhileNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "new while node failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyWhileNodeInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify while node input failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyWhileNodeExternalInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify while node external input failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyWhileNodeOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify while node output failed, ret:" << ret;
    return ret;
  }
  // update exit node user from exit to while
  ret = UpdateExitNodeUser();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "update while node users, ret:" << ret;
    return ret;
  }

  return ret;
}

// nodes between loop_cond op and merge op be added into cond_func_graph
STATUS FunctionalizeWhile::CondSubgraphAddNodes() {
  std::deque<AnfNodePtr> todo(512);
  todo.clear();
  for (size_t i = 1; i < loop_cond_node_->inputs().size(); i++) {
    todo.push_back(loop_cond_node_->input(i));
  }
  while (!todo.empty()) {
    AnfNodePtr node = todo.back();
    todo.pop_back();
    if (FunctionalizeControlOpPass::IsMerge(node)) {
      continue;
    }
    if (utils::isa<ParameterPtr>(node)) {
      cond_sub_func_graph_->add_parameter(node->cast<ParameterPtr>());
    } else {
      cond_sub_func_graph_->AddNode(node);
    }
    node->set_func_graph(cond_sub_func_graph_);
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        todo.push_back(cnode->input(i));
      }
    }
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyCondSubgraphInput() {
  std::vector<AnfNodePtr> nodes_need_drop{};
  for (auto &cnode : cond_sub_func_graph_->GetOrderedCnodes()) {
    for (auto &input_node : cnode->inputs()) {
      if (FunctionalizeControlOpPass::IsMerge(input_node)) {
        auto merge_node = input_node->cast<CNodePtr>();
        auto enter_node = BlongToWhichEnter(merge_node);
        int pos = PosInInputEnterNodes(enter_node);
        nodes_need_drop.push_back(cnode);

        // set parameter
        auto parameter = cond_sub_func_graph_->add_parameter();
        parameter->set_abstract(cnode->abstract());
        // hardcode for subgraph input name
        parameter->set_name(cond_subgraph_name_ + "_input_" + std::to_string(pos) + "_parameter");

        // replace merge
        auto manager = fg_->manager();
        auto node_users = manager->node_users()[cnode];
        for (auto &node_user : node_users) {
          if (cond_sub_func_graph_->nodes().contains(node_user.first)) {
            manager->SetEdge(node_user.first, node_user.second, parameter);
          }
        }
      }
    }
  }

  // drop node from cond_func_graph
  for (const auto &node : nodes_need_drop) {
    cond_sub_func_graph_->DropNode(node);
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyCondSubgraphOutput() {
  auto return_prim_ptr = std::make_shared<ops::Return>();
  if (return_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "GetReturnPrim return nullptr";
    return RET_NULL_PTR;
  }
  auto value_node = NewValueNode(return_prim_ptr);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "new value_node failed.";
    return RET_NULL_PTR;
  }
  // cond subgraph output is LoopCond's input
  std::vector<AnfNodePtr> op_inputs{value_node, loop_cond_node_->input(1)};
  auto return_cnode = cond_sub_func_graph_->NewCNode(op_inputs);
  return_cnode->set_fullname_with_scope(cond_subgraph_name_ + "-return");
  cond_sub_func_graph_->set_return(return_cnode);

  // hardcode subgraph outputs name
  cond_sub_func_graph_->output()->cast<CNodePtr>()->set_fullname_with_scope(cond_subgraph_name_ + "_output_0_cnode");
  return RET_OK;
}

STATUS FunctionalizeWhile::BuildCondGraph() {
  cond_subgraph_name_ = FunctionalizeControlOpPass::NodeClusterName(loop_cond_node_) + "_cond";
  cond_sub_func_graph_ =
    FunctionalizeControlOpPass::NewFuncGraph(cond_subgraph_name_, mindspore::lite::converter::FmkType_TF);
  if (cond_sub_func_graph_ == nullptr) {
    MS_LOG(ERROR) << "new cond_sub_func_graph_ return nullptr";
    return RET_NULL_PTR;
  }
  cond_sub_func_graph_->set_manager(fg_->manager());

  int ret = CondSubgraphAddNodes();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "add cond_subgraph node failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyCondSubgraphOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify cond_subgraph output failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyCondSubgraphInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify cond_subgraph input failed, ret:" << ret;
    return ret;
  }

  return ret;
}

// nodes between next_iteration op and switch op will be added into body_func_graph
STATUS FunctionalizeWhile::BodySubgraphAddNodes() {
  std::deque<AnfNodePtr> todo(512);
  todo.clear();
  for (auto &node : node_cluster_) {
    if (FunctionalizeControlOpPass::IsNextIteration(node)) {
      auto next_iteration_cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < next_iteration_cnode->inputs().size(); i++) {
        todo.push_back(next_iteration_cnode->input(i));
      }
      body_subgraph_output_map_[node] = next_iteration_cnode->input(1);
    }
  }

  while (!todo.empty()) {
    AnfNodePtr node = todo.back();
    todo.pop_back();
    if (FunctionalizeControlOpPass::IsSwitch(node)) {
      continue;
    }
    if (utils::isa<ParameterPtr>(node)) {
      body_sub_func_graph_->add_parameter(node->cast<ParameterPtr>());
    } else {
      body_sub_func_graph_->AddNode(node);
    }
    node->set_func_graph(body_sub_func_graph_);
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        todo.push_back(cnode->input(i));
      }
    }
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyBodySubgraphInput() {
  std::vector<AnfNodePtr> nodes_need_drop{};
  for (auto &cnode : body_sub_func_graph_->GetOrderedCnodes()) {
    for (auto &input_node : cnode->inputs()) {
      if (FunctionalizeControlOpPass::IsSwitch(input_node)) {
        auto switch_node = input_node->cast<CNodePtr>();
        auto merge_node = BlongToWhichMerge(switch_node);
        auto enter_node = BlongToWhichEnter(merge_node);
        int pos = PosInInputEnterNodes(enter_node);
        if (pos == POS_INVALID) {
          continue;
        }
        nodes_need_drop.push_back(cnode);

        // set parameter
        auto parameter = body_sub_func_graph_->add_parameter();
        parameter->set_abstract(cnode->abstract());
        // hardcode for subgraph input name
        parameter->set_name(body_subgraph_name_ + "_input_" + std::to_string(pos) + "_parameter");

        // replace switch
        auto manager = fg_->manager();
        auto node_users = manager->node_users()[cnode];
        for (auto &node_user : node_users) {
          if (body_sub_func_graph_->nodes().contains(node_user.first)) {
            manager->SetEdge(node_user.first, node_user.second, parameter);
          }
        }
      }
    }
  }

  // drop node from cond_func_graph
  for (const auto &node : nodes_need_drop) {
    body_sub_func_graph_->DropNode(node);
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::IdentifyBodySubgraphOutput() {
  std::vector<AnfNodePtr> tmp_output{};
  tmp_output.resize(input_enter_nodes_.size());

  for (auto &node_pair : body_subgraph_output_map_) {
    auto next_iteration_cnode = utils::cast<CNodePtr>(node_pair.first);
    auto switch_node = BlongToWhichSwitch(next_iteration_cnode);
    auto merge_node = BlongToWhichMerge(switch_node);
    auto enter_node = BlongToWhichEnter(merge_node);
    int pos = PosInInputEnterNodes(enter_node);
    if (pos == POS_INVALID) {
      continue;
    }

    tmp_output[pos] = node_pair.second;
    // hard code. set cnode output name
    node_pair.second->cast<CNodePtr>()->set_fullname_with_scope(body_subgraph_name_ + "_output_" + std::to_string(pos) +
                                                                "_cnode");
  }

  auto return_prim_ptr = std::make_shared<ops::Return>();
  if (return_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "GetReturnPrim return nullptr";
    return RET_NULL_PTR;
  }
  auto value_node = NewValueNode(return_prim_ptr);
  // cond subgraph output is LoopCond's input
  std::vector<AnfNodePtr> op_inputs{value_node};
  auto return_cnode = body_sub_func_graph_->NewCNode(op_inputs);
  return_cnode->set_fullname_with_scope(body_subgraph_name_ + "-return");

  if (tmp_output.size() == 1) {
    return_cnode->add_input(tmp_output[0]);
  } else {
    std::vector<AnfNodePtr> make_tuple_inputs = tmp_output;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetMakeTuplePrim return nullptr";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = body_sub_func_graph_->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope(return_cnode->fullname_with_scope() + "tuple");

    return_cnode->add_input(make_tuple_cnode);
  }

  body_sub_func_graph_->set_return(return_cnode);
  return RET_OK;
}

STATUS FunctionalizeWhile::BuildBodyGraph() {
  body_subgraph_name_ = FunctionalizeControlOpPass::NodeClusterName(loop_cond_node_) + "_body";
  body_sub_func_graph_ =
    FunctionalizeControlOpPass::NewFuncGraph(body_subgraph_name_, mindspore::lite::converter::FmkType_TF);
  if (body_sub_func_graph_ == nullptr) {
    MS_LOG(ERROR) << "new body_sub_func_graph_ return nullptr";
    return RET_NULL_PTR;
  }
  body_sub_func_graph_->set_manager(fg_->manager());

  int ret = BodySubgraphAddNodes();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "add body_subgraph node failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyBodySubgraphOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify body_subgraph output failed, ret:" << ret;
    return ret;
  }
  ret = IdentifyBodySubgraphInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "identify body_subgraph input failed, ret:" << ret;
    return ret;
  }
  return ret;
}

STATUS FunctionalizeWhile::InsertFuncGraphToWhileInput() {
  // set while input cond and body vnode
  auto cond_value_node = NewValueNode(cond_sub_func_graph_);
  auto body_value_node = NewValueNode(body_sub_func_graph_);
  auto inputs = while_node_->inputs();
  inputs.insert(inputs.begin() + 1, {cond_value_node, body_value_node});
  while_node_->set_inputs(inputs);
  return RET_OK;
}

STATUS FunctionalizeWhile::DropUselessNodesInMainGraph() {
  // fg_ drop cluster node
  for (auto &node : node_cluster_) {
    fg_->DropNode(node);
  }
  return RET_OK;
}

STATUS FunctionalizeWhile::Process() {
  int ret = BuildWhileNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "build while node failed, ret:" << ret;
    return ret;
  }

  ret = BuildCondGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "build condition graph failed, ret:" << ret;
    return ret;
  }

  ret = BuildBodyGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "build body graph failed, ret:" << ret;
    return ret;
  }

  ret = InsertFuncGraphToWhileInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "insert func_graph to while input failed, ret:" << ret;
    return ret;
  }

  ret = DropUselessNodesInMainGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "main func_graph drop nodes failed, ret:" << ret;
    return ret;
  }
  return ret;
}
}  // namespace mindspore::opt
