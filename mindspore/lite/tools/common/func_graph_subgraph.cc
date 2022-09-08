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

#define USE_DEPRECATED_API
#include "tools/common/func_graph_subgraph.h"
#include <set>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include "src/common/log_adapter.h"
#include "tools/common/node_util.h"
#include "tools/common/graph_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/partial_fusion.h"
#include "ops/core_ops.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
int SubGraph::Init(const std::set<CNodePtr> &head_nodes) {
  auto ret = InitSubGraphNode(head_nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphNode failed";
    return RET_ERROR;
  }
  ret = InitSubGraphInNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphInNode failed";
    return RET_ERROR;
  }
  ret = InitSubGraphOutNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphOutNode failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubGraph::Reset(const std::set<CNodePtr> &nodes, const std::set<CNodePtr> &head_nodes) {
  this->nodes_ = nodes;
  auto ret = InitSubGraphNode(head_nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphNode failed";
    return RET_ERROR;
  }
  ret = InitSubGraphInNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphInNode failed";
    return RET_ERROR;
  }
  ret = InitSubGraphOutNode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSubGraphOutNode failed";
    return RET_ERROR;
  }
  return RET_OK;
}

std::set<CNodePtr> SubGraph::GetNodes() const { return this->nodes_; }

std::set<CNodePtr> SubGraph::GetInCNodes() const { return this->in_nodes_; }

std::set<CNodePtr> SubGraph::GetInputCNodes() const {
  std::set<CNodePtr> inputs;
  for (const auto &in_node : in_nodes_) {
    if (in_node == nullptr) {
      continue;
    }
    auto input_cnodes = GetInputCNode(in_node);
    inputs.insert(input_cnodes.begin(), input_cnodes.end());
  }
  return inputs;
}

std::set<CNodePtr> SubGraph::GetOutCNodes() const { return this->out_nodes_; }

std::set<CNodePtr> SubGraph::FindCommonOutputs(const SubGraphPtr &subgraph) const {
  if (subgraph == nullptr) {
    return {};
  }
  std::set<CNodePtr> outputs_this = this->GetOutputCNodes();
  if (this == subgraph.get()) {
    return outputs_this;
  }
  std::set<CNodePtr> outputs_other = subgraph->GetOutputCNodes();
  std::set<CNodePtr> common_outputs;
  for (const auto &output1 : outputs_this) {
    if (output1 == nullptr) {
      continue;
    }
    auto iter = outputs_other.find(output1);
    if (iter == outputs_other.end()) {
      continue;
    }
    if (GetInputCNode(output1).size() == 2) {
      common_outputs.insert(output1);
    }
  }
  return common_outputs;
}

bool SubGraph::IfDependOnSameNode(const SubGraphPtr &subgraph) const {
  if (subgraph == nullptr || this == subgraph.get()) {
    return false;
  }
  std::set<CNodePtr> inputs_this = this->GetInputCNodes();
  std::set<CNodePtr> inputs_other = subgraph->GetInputCNodes();
  return std::any_of(inputs_this.begin(), inputs_this.end(), [&inputs_other](const CNodePtr &input_this) {
    if (input_this == nullptr) {
      return false;
    }
    return (inputs_other.count(input_this) > 0);
  });
}

std::set<CNodePtr> SubGraph::GetOutputCNodes() const {
  MS_ASSERT(belong_anf_ != nullptr);
  MS_ASSERT(belong_anf_->manager() != nullptr);
  auto node_users = belong_anf_->manager()->node_users();
  std::set<CNodePtr> outputs;
  for (const auto &out_node : out_nodes_) {
    if (out_node == nullptr) {
      continue;
    }
    auto iter = node_users.find(out_node);
    if (iter == node_users.end()) {
      continue;
    }
    auto post_node_pairs = iter->second;
    for (const auto &post_node_pair : post_node_pairs) {
      auto post_node = post_node_pair.first;
      if (post_node == nullptr || !utils::isa<CNodePtr>(post_node)) {
        continue;
      }
      outputs.insert(utils::cast<CNodePtr>(post_node));
    }
  }
  return outputs;
}

int SubGraph::InitSubGraphNode(const std::set<CNodePtr> &head_nodes) {
  MS_ASSERT(belong_anf_ != nullptr);
  MS_ASSERT(belong_anf_->manager() != nullptr);
  auto node_users = belong_anf_->manager()->node_users();
  std::queue<CNodePtr> q{};
  for (const auto &head_node : head_nodes) {
    if (head_node == nullptr) {
      continue;
    }
    q.push(head_node);
  }
  while (!q.empty()) {
    auto cur_node = q.front();
    MS_CHECK_TRUE_MSG(cur_node != nullptr, RET_NULL_PTR, "cur_node is nullptr");
    q.pop();
    this->nodes_.insert(cur_node);
    // check output-cnode of cur-node only depend on cur-node
    auto iter = node_users.find(cur_node);
    if (iter == node_users.end()) {
      continue;
    }
    auto post_node_pairs = iter->second;
    for (const auto &post_node_pair : post_node_pairs) {
      auto post_node = post_node_pair.first;
      if (post_node == nullptr || !utils::isa<CNodePtr>(post_node)) {
        continue;
      }
      auto post_cnode = utils::cast<CNodePtr>(post_node);
      MS_CHECK_TRUE_MSG(post_cnode != nullptr, RET_NULL_PTR, "cast failed");
      // return-node should not be include into subgraph absolutely // ut
      if (opt::CheckPrimitiveType(post_cnode, prim::kPrimReturn)) {
        continue;
      }
      MS_CHECK_TRUE_MSG(post_cnode != nullptr, RET_NULL_PTR, "post_cnode is nullptr");
      bool non_depend = true;
      // check all inputs of output-cnode
      for (const auto &input : post_cnode->inputs()) {
        if (input == nullptr) {
          continue;
        }
        // input cnode is not contained in subgraph
        if (utils::isa<CNodePtr>(input)) {
          auto input_cnode = utils::cast<CNodePtr>(input);
          MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_NULL_PTR, "cast ptr failed");
          if (this->nodes_.count(input_cnode) == 0) {
            non_depend = false;
            break;
          }
        }
        // input parameter is a graph input
        if (utils::isa<ParameterPtr>(input)) {
          auto input_parameter = utils::cast<ParameterPtr>(input);
          MS_CHECK_TRUE_MSG(input_parameter != nullptr, RET_NULL_PTR, "cast failed");
          if (!input_parameter->has_default()) {
            non_depend = false;
            break;
          }
        }
      }
      if (non_depend) {
        q.push(post_cnode);
      }
    }
  }
  return RET_OK;
}

int SubGraph::InitSubGraphInNode() {
  MS_ASSERT(belong_anf_ != nullptr);
  MS_ASSERT(belong_anf_->manager() != nullptr);
  auto node_users = belong_anf_->manager()->node_users();
  this->in_nodes_.clear();
  for (const auto &node : this->nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (std::any_of(node->inputs().begin(), node->inputs().end(), [this, &node_users](const auto &input) {
          if (input == nullptr) {
            return false;
          }
          if (utils::isa<CNodePtr>(input)) {
            auto input_cnode = utils::cast<CNodePtr>(input);
            MS_CHECK_TRUE_MSG(input_cnode != nullptr, false, "cast failed");
            if (this->nodes_.count(input_cnode) == 0) {
              return true;
            }
          }
          // graph input or shared weight input // ut
          if (utils::isa<ParameterPtr>(input)) {
            auto input_parameter = utils::cast<ParameterPtr>(input);
            MS_CHECK_TRUE_MSG(input_parameter != nullptr, false, "cast failed");
            if (!input_parameter->has_default()) {
              return true;
            }
            auto output_pair_iter = node_users.find(input);
            if (output_pair_iter != node_users.end() && output_pair_iter->second.size() > 1) {
              return true;
            }
          }
          return false;
        })) {
      in_nodes_.insert(node);
    }
  }
  return RET_OK;
}

int SubGraph::InitSubGraphOutNode() {
  MS_ASSERT(belong_anf_ != nullptr);
  MS_ASSERT(belong_anf_->manager() != nullptr);
  auto node_users = belong_anf_->manager()->node_users();
  this->out_nodes_.clear();
  for (const auto &node : this->nodes_) {
    if (node == nullptr) {
      continue;
    }
    auto node_users_iter = node_users.find(node);
    if (node_users_iter == node_users.end()) {
      continue;
    }
    auto node_output_pairs = node_users_iter->second;
    if (!std::any_of(node_output_pairs.begin(), node_output_pairs.end(),
                     [this](const std::pair<AnfNodePtr, int> &output_pair) {
                       auto output_node = output_pair.first;
                       if (output_node == nullptr || !utils::isa<CNodePtr>(output_node)) {
                         return false;
                       }
                       // graph output // ut
                       if (opt::CheckPrimitiveType(output_node, prim::kPrimReturn)) {
                         return true;
                       }
                       auto output_cnode = utils::cast<CNodePtr>(output_node);
                       MS_CHECK_TRUE_MSG(output_cnode != nullptr, false, "cast failed");
                       if (this->nodes_.count(output_cnode) == 0) {
                         return true;
                       }
                       return false;
                     }))
      continue;
    out_nodes_.insert(node);
  }
  return RET_OK;
}

bool SubGraph::MergeSubGraph(const SubGraphPtr &subgraph) {
  if (subgraph == nullptr || this == subgraph.get()) {
    return false;
  }
  // if two subgraph has same output, and this output node has only two input cnode which exactly from two
  // subgraph, we merge two subgraph, and find more post node
  auto common_outputs = this->FindCommonOutputs(subgraph);
  if (!common_outputs.empty()) {
    auto new_nodes = this->GetNodes();
    auto new_nodes2 = subgraph->GetNodes();
    new_nodes.insert(new_nodes2.begin(), new_nodes2.end());
    new_nodes.insert(common_outputs.begin(), common_outputs.end());
    if (this->Reset(new_nodes, common_outputs) != RET_OK) {
      MS_LOG(ERROR) << "Reset failed";
      return false;
    }
    return true;
  }

  if (this->IfDependOnSameNode(subgraph)) {
    auto new_nodes = this->GetNodes();
    auto new_nodes2 = subgraph->GetNodes();
    new_nodes.insert(new_nodes2.begin(), new_nodes2.end());
    if (this->Reset(new_nodes) != RET_OK) {
      MS_LOG(ERROR) << "Reset failed";
      return false;
    }
    return true;
  }
  return false;
}

// iterate node from in_nodes of current subgraph up to input of belong_anf
SubGraphPtr SubGraph::FindBeforeSubGraphInBelongAnf() const {
  MS_ASSERT(belong_anf_ == nullptr);
  // find before subgraph's nodes
  std::queue<CNodePtr> q{};
  std::set<CNodePtr> before_nodes{};
  for (const auto &node : this->GetInCNodes()) {
    for (const auto &in_cnode : lite::GetInputCNode(node)) {
      if (in_cnode == nullptr) {
        continue;
      }
      q.push(in_cnode);
    }
  }
  while (!q.empty()) {
    auto cur_cnode = q.front();
    MS_CHECK_TRUE_MSG(cur_cnode != nullptr, nullptr, "cur_cnode is nullptr");
    q.pop();
    before_nodes.insert(cur_cnode);
    for (const auto &in_cnode : lite::GetInputCNode(cur_cnode)) {
      q.push(in_cnode);
    }
  }
  // construct before subgraph
  auto before_subgraph = std::make_shared<SubGraph>(belong_anf_, this->name_ + "/before_subgraph");
  MS_CHECK_TRUE_MSG(before_subgraph != nullptr, nullptr, "before_subgraph is nullptr");
  if (before_subgraph->Reset(before_nodes) != RET_OK) {
    MS_LOG(ERROR) << "Reset failed";
    return nullptr;
  }
  return before_subgraph;
}

// iterate node from output of belong_anf up to out_nodes of current subgraph and before subgraph
SubGraphPtr SubGraph::FindAfterSubGraphInBelongAnf() const {
  MS_ASSERT(belong_anf_ == nullptr);
  // find before subgraph
  auto before_subgraph = this->FindBeforeSubGraphInBelongAnf();
  if (before_subgraph == nullptr) {
    MS_LOG(ERROR) << "Find before subgraph failed";
    return nullptr;
  }
  // find after subgraph's nodes
  std::queue<CNodePtr> q{};
  std::set<CNodePtr> after_nodes{};
  auto output_node = belong_anf_->output();
  if (!utils::isa<CNodePtr>(output_node)) {
    MS_LOG(ERROR) << "Output node of anf should be a cnode: " << output_node->fullname_with_scope();
    return nullptr;
  }
  q.push(utils::cast<CNodePtr>(output_node));
  auto subgraph_out_nodes = this->GetOutCNodes();
  auto before_out_nodes = before_subgraph->GetOutCNodes();
  while (!q.empty()) {
    auto cur_cnode = q.front();
    MS_CHECK_TRUE_MSG(cur_cnode != nullptr, nullptr, "cur_cnode is nullptr");
    q.pop();
    after_nodes.insert(cur_cnode);
    for (const auto &in_cnode : lite::GetInputCNode(cur_cnode)) {
      if (subgraph_out_nodes.count(in_cnode) == 0 && before_out_nodes.count(in_cnode) == 0) {
        q.push(in_cnode);
      }
    }
  }
  // construct before subgraph
  auto after_subgraph = std::make_shared<SubGraph>(belong_anf_, this->name_ + "/after_subgraph");
  MS_CHECK_TRUE_MSG(after_subgraph != nullptr, nullptr, "after_subgraph is nullptr");
  if (after_subgraph->Reset(after_nodes) != RET_OK) {
    MS_LOG(ERROR) << "Reset failed";
    return nullptr;
  }

  return after_subgraph;
}

int SubGraph::CreatePartialInBelongAnf() {
  MS_ASSERT(this->belong_anf_ != nullptr);
  MS_ASSERT(this->belong_anf_->manager() != nullptr);
  // determine func_graph name
  std::string graph_name = this->name_;
  if (graph_name.empty()) {
    if (this->nodes_.empty()) {
      graph_name = "subgraph";
    } else {
      graph_name = (*(this->nodes_.begin()))->fullname_with_scope() + "/subgraph";
    }
  }
  // create func_graph of partial
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_NULL_PTR, "func_graph is nullptr");
  auto manager = belong_anf_->manager();
  manager->AddFuncGraph(func_graph);
  func_graph->set_attr("graph_name", MakeValue(graph_name));
  func_graph->set_manager(manager);
  // create cnode and parameter for func_graph of partial
  std::vector<AnfNodePtr> partial_inputs;
  std::map<AnfNodePtr, AnfNodePtr> partial_inputs_and_subgraph_input_map;
  auto ret = CreateParameterForPartialSubGraph(func_graph, &partial_inputs, &partial_inputs_and_subgraph_input_map);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "CreateParameterForPartialSubGraph  failed";
    return ret;
  }
  ret = CreateCNodeForPartialSubGraph(func_graph, partial_inputs_and_subgraph_input_map);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "CreateCNodeForPartialSubGraph failed";
    return ret;
  }
  // add return for func_graph of partial
  auto sub_graph_outputs = this->GetOutCNodes();
  MS_ASSERT(!sub_graph_outputs.empty());
  ret = SetFuncGraphOutput(func_graph, sub_graph_outputs);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "Set subgraph output failed";
    return ret;
  }
  // create partial cnode
  auto partial_prim = std::make_shared<mindspore::ops::PartialFusion>();
  auto graph_value_node = NewValueNode(func_graph);
  MS_CHECK_TRUE_MSG(partial_prim != nullptr, RET_NULL_PTR, "partial_prim is nullptr");
  MS_CHECK_TRUE_MSG(graph_value_node != nullptr, RET_NULL_PTR, "graph_value_node is nullptr");
  auto partial_prim_c = partial_prim->GetPrim();
  MS_CHECK_TRUE_MSG(partial_prim_c != nullptr, RET_NULL_PTR, "partial_prim_c is nullptr");
  partial_inputs.insert(partial_inputs.begin(), graph_value_node);
  auto partial_cnode = belong_anf_->NewCNode(partial_prim_c, partial_inputs);
  MS_CHECK_TRUE_MSG(partial_cnode != nullptr, RET_NULL_PTR, "partial_cnode is nullptr");
  partial_cnode->set_fullname_with_scope(graph_name + "/partial");
  for (size_t i = 0; i < partial_inputs.size(); ++i) {
    const auto &input = partial_inputs.at(i);
    manager->SetEdge(partial_cnode, static_cast<int>(i + 1), input);
  }
  // create call cnode
  std::vector<AnfNodePtr> call_node_inputs{partial_cnode};
  auto call_cnode = belong_anf_->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(call_cnode != nullptr, RET_NULL_PTR, "call_cnode is nullptr");
  call_cnode->set_fullname_with_scope(graph_name + "/call");
  // replace belong-graph's output
  auto return_node = belong_anf_->get_return();
  // return node should has 2 inputs
  MS_ASSERT(return_node != nullptr && return_node->inputs().size() == 2);
  auto ori_output = return_node->inputs().at(1);
  manager->Replace(ori_output, call_cnode);
  return RET_OK;
}

int SubGraph::SetFuncGraphOutput(const FuncGraphPtr &graph, const std::set<CNodePtr> &outputs) {
  std::vector<AnfNodePtr> output_nodes;
  output_nodes.insert(output_nodes.end(), outputs.begin(), outputs.end());
  return lite::SetFuncGraphOutput(graph, output_nodes);
}

int SubGraph::CreateParameterForPartialSubGraph(
  const FuncGraphPtr &sub_graph, std::vector<AnfNodePtr> *partial_inputs,
  std::map<AnfNodePtr, AnfNodePtr> *partial_inputs_and_subgraph_input_map) {
  MS_ASSERT(sub_graph != nullptr);
  MS_ASSERT(partial_inputs != nullptr && partial_inputs->empty());
  MS_ASSERT(partial_inputs_and_subgraph_input_map != nullptr && partial_inputs_and_subgraph_input_map->empty());
  MS_CHECK_TRUE_MSG(sub_graph->get_attr("graph_name") != nullptr, RET_ERROR, "graph_name is nullptr");
  std::string graph_name = sub_graph->get_attr("graph_name")->ToString();
  for (const auto &in_cnode : this->GetInCNodes()) {
    if (in_cnode == nullptr) {
      continue;
    }
    for (size_t i = 1; i < in_cnode->inputs().size(); i++) {
      auto input = in_cnode->input(i);
      if (input == nullptr) {
        continue;
      }
      auto iter = partial_inputs_and_subgraph_input_map->find(input);
      if (iter != partial_inputs_and_subgraph_input_map->end()) {
        continue;
      }
      // create subgraph input parameter from cnode and record partial inputs
      if (utils::isa<CNodePtr>(input)) {
        auto input_cnode = utils::cast<CNodePtr>(input);
        MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_NULL_PTR, "cast ptr failed");
        if (this->GetNodes().count(input_cnode) > 0) {
          continue;
        }
        partial_inputs->emplace_back(input);
        auto new_parameter = sub_graph->add_parameter();
        new_parameter->set_name(graph_name + "_input_" + input->fullname_with_scope());
        new_parameter->set_abstract(input->abstract());
        (*partial_inputs_and_subgraph_input_map)[input] = new_parameter;
      }
      // create subgraph input parameter from parameter and record partial inputs
      // add parameter to func_graph
      auto node_users = this->belong_anf_->manager()->node_users();
      if (utils::isa<ParameterPtr>(input)) {
        auto parameter = utils::cast<ParameterPtr>(input);
        MS_CHECK_TRUE_MSG(parameter != nullptr, RET_NULL_PTR, "cast ptr failed");
        // graph input: create a parameter
        if (!parameter->has_default()) {
          auto new_parameter = sub_graph->add_parameter();
          new_parameter->set_name(graph_name + "_input_" + input->fullname_with_scope());
          new_parameter->set_abstract(input->abstract());
          (*partial_inputs_and_subgraph_input_map)[input] = new_parameter;
          partial_inputs->emplace_back(new_parameter);
        }
        // weight parameter, it depends
        auto output_pairs_iter = node_users.find(input);
        if (output_pairs_iter != node_users.end() &&
            output_pairs_iter->second.size() > 1) {  // shared weight: create a parameter
          auto new_parameter = sub_graph->add_parameter();
          new_parameter->set_name(graph_name + "_input_" + input->fullname_with_scope());
          new_parameter->set_abstract(input->abstract());
          (*partial_inputs_and_subgraph_input_map)[input] = new_parameter;
          partial_inputs->emplace_back(new_parameter);
        } else {  // not shared weight: move into subgraph
          sub_graph->AddNode(input);
          input->set_func_graph(sub_graph);
          this->belong_anf_->DropNode(input);
        }
      }
    }
  }
  return RET_OK;
}

int SubGraph::CreateCNodeForPartialSubGraph(
  const FuncGraphPtr &sub_graph, const std::map<AnfNodePtr, AnfNodePtr> &partial_inputs_and_subgraph_input_map) {
  MS_ASSERT(sub_graph != nullptr);
  // move cnode from belong_graph to subgraph
  for (auto &node : this->GetNodes()) {
    sub_graph->AddNode(node);
    if (!utils::isa<ValueNodePtr>(node)) {
      node->set_func_graph(sub_graph);
    }
    for (size_t i = 0; i < node->inputs().size(); i++) {
      auto input = node->inputs().at(i);
      if (input == nullptr) {
        continue;
      }
      auto iter = partial_inputs_and_subgraph_input_map.find(input);
      if (iter == partial_inputs_and_subgraph_input_map.end()) {
        continue;
      }
      // use SetEdge not set_input, if not, node_user is not updated.
      this->belong_anf_->manager()->SetEdge(node, static_cast<int>(i), iter->second);
    }
    this->belong_anf_->DropNode(node);
  }
  return RET_OK;
}

int SubGraph::ApplySubGraph() {
  // check
  if (this->nodes_.empty()) {
    return lite::RET_NO_CHANGE;
  }
  if (belong_anf_ == nullptr || belong_anf_->manager() == nullptr) {
    MS_LOG(DEBUG) << "belong_anf_ or manager is nullptr";
    return lite::RET_NO_CHANGE;
  }
  for (const auto &node : this->nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (node->func_graph() != belong_anf_) {
      MS_LOG(DEBUG) << "subgraph nodes belong to different func_graph";
      return lite::RET_ERROR;
    }
  }

  // create after partial // redirect input of after subgraph
  auto after_subgraph = this->FindAfterSubGraphInBelongAnf();
  if (after_subgraph == nullptr) {
    MS_LOG(DEBUG) << "Create after subgraph failed";
    return RET_ERROR;
  }
  auto ret = after_subgraph->CreatePartialInBelongAnf();
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "Create after partial failed";
    return RET_ERROR;
  }
  // merge after partial into subgraph
  auto subgraph_nodes = this->nodes_;
  auto return_node = belong_anf_->get_return();
  MS_ASSERT(return_node != nullptr && return_node->inputs().size() == 2);
  auto call_node = return_node->inputs().at(1);
  MS_ASSERT(call_node != nullptr && utils::isa<CNodePtr>(call_node));
  auto call_cnode = utils::cast<CNodePtr>(call_node);
  MS_ASSERT(call_cnode != nullptr && call_cnode->inputs().size() == 1);
  auto after_partial_node = call_cnode->inputs().at(0);
  MS_ASSERT(after_partial_node != nullptr && utils::isa<CNodePtr>(after_partial));
  auto after_partial_cnode = utils::cast<CNodePtr>(after_partial_node);
  MS_ASSERT(after_partial_cnode != nullptr);
  subgraph_nodes.insert(after_partial_cnode);
  subgraph_nodes.insert(call_cnode);
  if (this->Reset(subgraph_nodes) != RET_OK) {
    MS_LOG(ERROR) << "Reset failed";
    return RET_ERROR;
  }
  // create subgraph partial // add partial to main subgraph
  ret = this->CreatePartialInBelongAnf();
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "Create partial failed";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
