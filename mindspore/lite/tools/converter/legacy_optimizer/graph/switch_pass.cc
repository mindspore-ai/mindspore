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

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include "tools/converter/legacy_optimizer/graph/switch_pass.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"

namespace mindspore::lite {

STATUS SwitchPass::Run(mindspore::schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    auto type = node->primitive->value.type;
    if (type != schema::PrimitiveType_Switch) {
      continue;
    }

    SingleSwitchPass pass(graph, i);
    int ret = pass.Run();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "node: " << node->name << "'s switch pass failed: " << ret;
      return ret;
    }
  }
  // remove empty subgraphs
  std::vector<std::unique_ptr<SubGraphT>> new_sub_graphs;
  std::map<uint32_t, uint32_t> sub_graph_index_map;
  for (size_t i = 0; i < graph->subGraph.size(); ++i) {
    auto &sub_graph = graph->subGraph.at(i);
    if (!sub_graph->nodeIndices.empty()) {
      new_sub_graphs.emplace_back(std::move(sub_graph));
      sub_graph_index_map.emplace(std::make_pair(i, new_sub_graphs.size() - 1));
    }
  }
  graph->subGraph.swap(new_sub_graphs);
  for (size_t i = 0; i < graph->nodes.size(); ++i) {
    auto &node = graph->nodes.at(i);
    auto type = node->primitive->value.type;
    if (type != schema::PrimitiveType_PartialFusion) {
      continue;
    }
    MS_ASSERT(node->primitive != nullptr);
    MS_ASSERT(node->primitive->value..AsPartialFusion() != nullptr);
    auto partial_prim = node->primitive->value.AsPartialFusion();
    if (partial_prim->sub_graph_index == -1) {
      continue;
    }
    if (sub_graph_index_map.find(partial_prim->sub_graph_index) == sub_graph_index_map.end()) {
      MS_LOG(ERROR) << "sub_graph_index is illegal";
      return RET_ERROR;
    }
    partial_prim->sub_graph_index = sub_graph_index_map[partial_prim->sub_graph_index];
  }
  return RET_OK;
}

STATUS SingleSwitchPass::DoubleSwitchOutput() {
  auto cur_switch_output_tensor_indices = switch_node_->outputIndex;
  if (cur_switch_output_tensor_indices.size() != first_partial_node_->inputIndex.size()) {
    MS_LOG(ERROR) << "switch node: " << switch_node_->name << " input or output number is not right.";
    return RET_ERROR;
  }
  MS_ASSERT(origin_switch_output_tensor_indices_.size() == first_partial_node_->inputIndex.szie());
  for (size_t i = 0; i < cur_switch_output_tensor_indices.size(); i++) {
    auto &switch_out_tensor = graph_->allTensors.at(cur_switch_output_tensor_indices[i]);
    const auto &cond_partial_input_tensor = graph_->allTensors.at(first_partial_node_->inputIndex[i]);
    switch_out_tensor->dataType = cond_partial_input_tensor->dataType;
    auto tensor = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    switch_node_->outputIndex.push_back(graph_->allTensors.size() - 1);
    graph_->subGraph.at(this_subgraph_index_)->tensorIndices.push_back(graph_->allTensors.size() - 1);
  }
  return RET_OK;
}

void SingleSwitchPass::UpdateSwitchOutputIndices(uint32_t *idx) {
  auto iter = std::find(switch_node_->outputIndex.begin(), switch_node_->outputIndex.end(), *idx);
  if (iter != switch_node_->outputIndex.end()) {
    int pos = iter - switch_node_->outputIndex.begin();
    *idx = switch_node_->outputIndex.at(pos + switch_node_->outputIndex.size() / 2);
  }
}

STATUS SingleSwitchPass::UpdateSwitchUser() {
  for (auto &node_idx : graph_->subGraph.at(this_subgraph_index_)->nodeIndices) {
    auto &node = graph_->nodes.at(node_idx);
    for (auto &idx : node->inputIndex) {
      UpdateSwitchOutputIndices(&idx);
    }
  }
  // update graph switch user
  for (auto &subgraph : graph_->subGraph) {
    for (auto &idx : subgraph->outputIndices) {
      UpdateSwitchOutputIndices(&idx);
    }
  }

  for (auto &idx : graph_->outputIndex) {
    UpdateSwitchOutputIndices(&idx);
  }

  return RET_OK;
}

bool SingleSwitchPass::IsLoop() {
  for (auto &node : second_graph_nodes_) {
    if (node->primitive->value.type == schema::PrimitiveType_PartialFusion &&
        node->primitive->value.AsPartialFusion() != nullptr &&
        node->primitive->value.AsPartialFusion()->sub_graph_index == first_subgraph_index_) {
      body_to_cond_partial_node_ = node;
      return true;
    }
  }
  return false;
}

std::unique_ptr<schema::TensorT> SingleSwitchPass::NewTensor(const std::unique_ptr<schema::TensorT> &in_tensor,
                                                             bool with_data) {
  auto out_tensor = std::make_unique<schema::TensorT>();
  out_tensor->nodeType = in_tensor->nodeType;
  out_tensor->dims = in_tensor->dims;
  out_tensor->dataType = in_tensor->dataType;
  out_tensor->format = in_tensor->format;
  if (with_data) {
    out_tensor->data = in_tensor->data;
  }
  return out_tensor;
}

STATUS SingleSwitchPass::BodyGraphVariableInput(std::vector<size_t> *variable_input) {
  auto &body_fg = graph_->subGraph.at(second_subgraph_index_);
  auto body_fg_output = body_fg->outputIndices;
  for (auto &subgraph_output : body_fg_output) {
    for (auto &node : second_graph_nodes_) {
      if (node != nullptr && IsContain(node->outputIndex, subgraph_output)) {
        int partial_idx = GetSubgraphOutputTensorIndex(body_fg, node);
        if (partial_idx == -1) {
          MS_LOG(ERROR) << "get input index failed.";
          return RET_ERROR;
        }
        (*variable_input).emplace_back(partial_idx);
      }
    }
  }
  return RET_OK;
}

std::unique_ptr<schema::CNodeT> SingleSwitchPass::MakeMergeNode(const std::string &name,
                                                                const std::vector<size_t> &const_input) {
  auto merge_node = std::make_unique<schema::CNodeT>();
  if (merge_node == nullptr) {
    MS_LOG(ERROR) << "new CNodeT failed";
    return nullptr;
  }
  merge_node->primitive = std::make_unique<PrimitiveT>();
  if (merge_node->primitive == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }

  merge_node->name = name;
  merge_node->primitive->value.type = schema::PrimitiveType_Merge;
  merge_node->primitive->value.value = new (std::nothrow) MergeT();
  if (merge_node->primitive->value.value == nullptr) {
    MS_LOG(ERROR) << "new MergeT failed";
    return nullptr;
  }

  // merge node output is same as switch
  for (auto &out_index : origin_switch_output_tensor_indices_) {
    auto &switch_out_tensor = graph_->allTensors.at(out_index);
    auto tensor = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    merge_node->outputIndex.push_back(graph_->allTensors.size() - 1);
  }

  merge_node->inputIndex.assign(first_partial_node_->inputIndex.begin(), first_partial_node_->inputIndex.end());

  std::set<uint32_t> input_set{};
  for (auto &iter : merge_node->inputIndex) {
    if (input_set.find(iter) != input_set.end()) {
      auto &in_tensor = graph_->allTensors.at(iter);
      auto tensor = NewTensor(in_tensor, true);
      graph_->allTensors.push_back(std::move(tensor));
      iter = graph_->allTensors.size() - 1;
    }
    input_set.insert(iter);
  }

  // double merge inputs to contain the outputs of body  node
  auto old_merge_input = merge_node->inputIndex;
  for (size_t i = 0; i < old_merge_input.size(); i++) {
    auto &in_tensor = graph_->allTensors.at(old_merge_input[i]);
    if (IsContain(const_input, i)) {
      merge_node->inputIndex.push_back(old_merge_input[i]);
    } else {
      auto tensor = NewTensor(in_tensor);
      tensor->nodeType = NodeType_CNode;
      graph_->allTensors.push_back(std::move(tensor));
      merge_node->inputIndex.push_back(graph_->allTensors.size() - 1);
    }
  }
  return merge_node;
}

STATUS SingleSwitchPass::InsertMerge() {
  // update body graph output
  auto &body_fg = graph_->subGraph.at(second_subgraph_index_);
  body_fg->outputIndices.assign(body_to_cond_partial_node_->inputIndex.begin(),
                                body_to_cond_partial_node_->inputIndex.end());

  // remove body_to_cond_partial_node_ from second_graph_nodes_
  for (auto it = second_graph_nodes_.begin(); it != second_graph_nodes_.end();) {
    if (*it == body_to_cond_partial_node_) {
      it = second_graph_nodes_.erase(it);
    } else {
      it++;
    }
  }

  // isolate body_to_cond_partial_node_
  IsolateUselessNode(body_to_cond_partial_node_, graph_);

  std::vector<size_t> variable_input{};
  int ret = BodyGraphVariableInput(&variable_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "get body graph variable input failed, ret: " << ret;
    return ret;
  }

  std::vector<size_t> const_input{};
  for (size_t i = 0; i < second_partial_node_->inputIndex.size(); i++) {
    if (IsContain(variable_input, i)) {
      continue;
    }
    const_input.push_back(i);
  }
  auto merge_node = MakeMergeNode(switch_node_->name + "-merge", const_input);
  if (merge_node == nullptr) {
    MS_LOG(ERROR) << "make merge node failed";
    return ret;
  }
  // insert merge node before the cond graph
  std::map<int, int> cond_input_update_map{};
  for (size_t i = 0; i < first_partial_node_->inputIndex.size(); i++) {
    cond_input_update_map.insert(std::make_pair(first_partial_node_->inputIndex.at(i), merge_node->outputIndex.at(i)));
  }
  for (auto &node : first_graph_nodes_) {
    for (auto &input_idx : node->inputIndex) {
      if (cond_input_update_map.find(input_idx) != cond_input_update_map.end()) {
        input_idx = cond_input_update_map.at(input_idx);
      }
    }
  }

  // update cond node input to be consistent with cond graph input
  first_partial_node_->inputIndex.assign(merge_node->outputIndex.begin(), merge_node->outputIndex.end());

  // insert switch after cond node and merge node
  auto cond_input = switch_node_->inputIndex.front();
  switch_node_->inputIndex.clear();
  switch_node_->inputIndex.push_back(cond_input);
  switch_node_->inputIndex.insert(switch_node_->inputIndex.end(), merge_node->outputIndex.begin(),
                                  merge_node->outputIndex.end());

  // move body node to switch node output
  second_partial_node_->inputIndex.clear();
  second_partial_node_->inputIndex.assign(switch_node_->outputIndex.begin(),
                                          switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2);

  // skip tensor which is not any nodes' inputs to avoid body partial connect to merge input cnode
  std::vector<uint32_t> skip_input_tensors;
  for (auto input : const_input) {
    auto real_input = graph_->subGraph.at(second_subgraph_index_)->inputIndices.at(input);
    bool skip = true;
    for (auto &node : second_graph_nodes_) {
      if (IsContain(node->inputIndex, real_input)) {
        skip = false;
        break;
      }
    }
    if (skip) {
      auto &skip_tensor = graph_->allTensors.at(real_input);
      int partial_idx = GetSubgraphInputTensorIndex(graph_->subGraph.at(second_subgraph_index_), skip_tensor);
      skip_input_tensors.emplace_back(partial_idx);
    }
  }

  // concat body output to merge input
  second_partial_node_->outputIndex.clear();
  for (uint32_t merge_right_input = 0; merge_right_input < merge_node->inputIndex.size() / 2; merge_right_input++) {
    if (!IsContain(skip_input_tensors, merge_right_input)) {
      second_partial_node_->outputIndex.emplace_back(
        merge_node->inputIndex.at(merge_node->inputIndex.size() / 2 + merge_right_input));
    } else {
      second_partial_node_->outputIndex.emplace_back(UINT32_MAX);
    }
  }

  graph_->nodes.push_back(std::move(merge_node));

  return RET_OK;
}

STATUS SingleSwitchPass::InsertPartialAndMergeAfterSwitch() {
  // insert partial

  // origin switch node in : T partial | F partial | condition node | partial inputs...
  // origin switch node out : partial outputs
  // converted switch node in : condition node | partial inputs...
  // converted switch node out : double partial inputs...

  first_partial_node_->outputIndex.clear();
  second_partial_node_->outputIndex.clear();
  for (auto &out_index : switch_node_->outputIndex) {
    auto &switch_out_tensor = graph_->allTensors.at(out_index);
    auto tensor1 = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor1));
    first_partial_node_->outputIndex.push_back(graph_->allTensors.size() - 1);
    auto tensor2 = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor2));
    second_partial_node_->outputIndex.push_back(graph_->allTensors.size() - 1);
  }

  auto origin_switch_outputs = switch_node_->outputIndex;
  switch_node_->outputIndex.clear();
  for (size_t i = 3; i < switch_node_->inputIndex.size(); i++) {
    auto &switch_in_tensor = graph_->allTensors.at(switch_node_->inputIndex[i]);
    auto tensor = NewTensor(switch_in_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    switch_node_->outputIndex.push_back(graph_->allTensors.size() - 1);
  }
  int ret = DoubleSwitchOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Double switch outputs failed";
    return ret;
  }

  switch_node_->inputIndex.erase(switch_node_->inputIndex.begin(), switch_node_->inputIndex.begin() + 2);
  MS_ASSERT(switch_node_->outputIndex.size() % 2 == 0);
  first_partial_node_->inputIndex.assign(switch_node_->outputIndex.begin(),
                                         switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2);
  second_partial_node_->inputIndex.assign(switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2,
                                          switch_node_->outputIndex.end());

  // insert merge
  auto merge_node = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
  if (merge_node == nullptr) {
    MS_LOG(ERROR) << "new cnode failed";
    return RET_NULL_PTR;
  }
  merge_node->primitive = std::unique_ptr<PrimitiveT>(new (std::nothrow) PrimitiveT);
  if (merge_node->primitive == nullptr) {
    MS_LOG(ERROR) << "new primitiveT failed";
    return RET_NULL_PTR;
  }
  merge_node->name = switch_node_->name + "-merge";
  merge_node->primitive->value.type = schema::PrimitiveType_Merge;
  merge_node->primitive->value.value = new (std::nothrow) MergeT();
  if (merge_node->primitive->value.value == nullptr) {
    MS_LOG(ERROR) << "new MergeT failed";
    return RET_NULL_PTR;
  }
  if (first_graph_nodes_.empty()) {
    merge_node->inputIndex.assign(switch_node_->outputIndex.begin(),
                                  switch_node_->outputIndex.begin() + first_partial_node_->outputIndex.size());
    first_subgraph_index_ = -1;
    IsolateUselessNode(first_partial_node_, graph_);
  } else {
    merge_node->inputIndex.assign(first_partial_node_->outputIndex.begin(), first_partial_node_->outputIndex.end());
  }

  if (second_graph_nodes_.empty()) {
    merge_node->inputIndex.insert(merge_node->inputIndex.end(),
                                  switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2,
                                  switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2 +
                                    second_partial_node_->outputIndex.size());
    second_subgraph_index_ = -1;
    IsolateUselessNode(second_partial_node_, graph_);
  } else {
    merge_node->inputIndex.insert(merge_node->inputIndex.end(), second_partial_node_->outputIndex.begin(),
                                  second_partial_node_->outputIndex.end());
  }
  merge_node->outputIndex = origin_switch_outputs;
  graph_->nodes.push_back(std::move(merge_node));
  return RET_OK;
}

void SingleSwitchPass::IsolateUselessNode(schema::CNodeT *partial_node, schema::MetaGraphT *graph) {
  partial_node->inputIndex.clear();
  partial_node->outputIndex.clear();
}

size_t SingleSwitchPass::InitThisGraphIndex() {
  for (size_t i = 0; i < graph_->subGraph.size(); i++) {
    if (std::any_of(graph_->subGraph.at(i)->nodeIndices.begin(), graph_->subGraph.at(i)->nodeIndices.end(),
                    [this](const uint32_t &idx) { return idx == this->switch_node_index_; })) {
      return i;
    }
  }
  return -1;
}

STATUS SingleSwitchPass::Init() {
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return RET_NULL_PTR;
  }

  this_subgraph_index_ = InitThisGraphIndex();
  if (this_subgraph_index_ < 0) {
    MS_LOG(ERROR) << "init this subgraph index failed.";
    return RET_ERROR;
  }

  switch_node_ = graph_->nodes.at(switch_node_index_).get();
  if (switch_node_ == nullptr) {
    MS_LOG(ERROR) << "switch node is nullptr.";
    return RET_NULL_PTR;
  }

  if (switch_node_->inputIndex.size() < kSwitchMinInputSize) {
    MS_LOG(ERROR) << "switch node: " << switch_node_->name
                  << " 's input size is not right, size: " << switch_node_->inputIndex.size();
    return RET_INPUT_PARAM_INVALID;
  }

  origin_switch_output_tensor_indices_ = switch_node_->outputIndex;

  // get cond_partial_node_ and second_partial_node_
  bool find_cond_node = false;
  bool find_body_node = false;
  for (auto iter = graph_->nodes.begin(); iter < graph_->nodes.end(); iter++) {
    for (auto &out_index : iter->get()->outputIndex) {
      if (out_index == switch_node_->inputIndex[kSwitchFirstIndex]) {
        first_partial_node_ = iter->get();
        find_cond_node = true;
      }
      if (out_index == switch_node_->inputIndex[kSwitchSecondIndex]) {
        second_partial_node_ = iter->get();
        find_body_node = true;
      }
    }
    if (find_body_node && find_cond_node) {
      break;
    }
  }

  // get cond_graph_nodes_
  MS_ASSERT(first_partial_node_->primitive->value..AsPartialFusion() != nullptr);
  first_subgraph_index_ = first_partial_node_->primitive->value.AsPartialFusion()->sub_graph_index;
  auto cond_node_indices = graph_->subGraph.at(first_subgraph_index_)->nodeIndices;
  for (auto &index : cond_node_indices) {
    first_graph_nodes_.push_back(graph_->nodes.at(index).get());
  }

  // get second_graph_nodes_
  MS_ASSERT(second_partial_node_->primitive->value..AsPartialFusion() != nullptr);
  second_subgraph_index_ = second_partial_node_->primitive->value.AsPartialFusion()->sub_graph_index;
  auto body_node_indices = graph_->subGraph.at(second_subgraph_index_)->nodeIndices;
  for (auto &index : body_node_indices) {
    second_graph_nodes_.push_back(graph_->nodes.at(index).get());
  }

  // get this_graph_nodes_
  auto this_node_indices = graph_->subGraph.at(this_subgraph_index_)->nodeIndices;
  for (auto &index : this_node_indices) {
    this_graph_nodes_.push_back(graph_->nodes.at(index).get());
  }
  return RET_OK;
}

int SingleSwitchPass::GetSubgraphInputTensorIndex(const std::unique_ptr<SubGraphT> &subgraph,
                                                  const std::unique_ptr<TensorT> &tensor) {
  int partial_idx = -1;
  if (tensor->name.find("_input_") != std::string::npos) {
    // get parameter input index k. subgraph name + “_input_" + "k"
    auto pos = subgraph->name.size() + sizeof("_input_");
    auto pos2 = tensor->name.find('_', pos);
    auto idx_str = tensor->name.substr(pos - 1, pos2 - pos + 1);
    partial_idx = std::stoi(idx_str);
  }

  if (tensor->name.find("_output_") != std::string::npos) {
    // get parameter input index k. subgraph name + “_output_" + "k"
    auto pos = subgraph->name.size() + sizeof("_output_");
    auto pos2 = tensor->name.find('_', pos);
    auto idx_str = tensor->name.substr(pos - 1, pos2 - pos + 1);
    partial_idx = std::stoi(idx_str);
  }
  return partial_idx;
}

int SingleSwitchPass::GetSubgraphOutputTensorIndex(const std::unique_ptr<SubGraphT> &subgraph, CNodeT *node) {
  int partial_idx = -1;
  if (node->name == "LogicalAnd") {
    partial_idx = 0;
  } else {
    // get parameter input index k. subgraph name + “_output_" + "k"
    auto pos = subgraph->name.size() + sizeof("_output_");
    auto pos2 = node->name.find('_', pos);
    auto idx_str = node->name.substr(pos - 1, pos2 - pos + 1);
    partial_idx = std::stoi(idx_str);
  }
  return partial_idx;
}

STATUS SingleSwitchPass::UpdateSubgraphInput(const size_t &subgraph_index, schema::CNodeT *partial_node,
                                             const std::vector<schema::CNodeT *> &subgraph_nodes) {
  if (partial_node == nullptr) {
    MS_LOG(ERROR) << "partial_node is nullptr or subgraph_nodes are empty.";
    return RET_INPUT_PARAM_INVALID;
  }
  auto &partial_inputs = partial_node->inputIndex;
  auto &subgraph = graph_->subGraph.at(subgraph_index);
  auto &subgraph_inputs = subgraph->inputIndices;

  std::map<int, int> subgraph_input_map;
  std::vector<std::pair<int, int>> tmp_inputs_order{};
  for (unsigned int &subgraph_input : subgraph_inputs) {
    auto &tensor = graph_->allTensors.at(subgraph_input);
    int partial_idx = GetSubgraphInputTensorIndex(subgraph, tensor);
    if (partial_idx == -1) {
      MS_LOG(ERROR) << "get input index failed.";
      return RET_ERROR;
    }
    subgraph_input_map.insert(std::pair<int, int>{subgraph_input, partial_inputs[partial_idx]});
    tmp_inputs_order.emplace_back(partial_idx, partial_inputs[partial_idx]);
  }

  for (auto &subgraph_node : subgraph_nodes) {
    for (auto &input : subgraph_node->inputIndex) {
      if (subgraph_input_map.find(input) != subgraph_input_map.end()) {
        input = subgraph_input_map.at(input);
      }
    }
  }

  std::sort(tmp_inputs_order.begin(), tmp_inputs_order.end(),
            [](std::pair<int, int> a, std::pair<int, int> b) { return a.first < b.first; });

  std::vector<int> new_subgraph_inputs{};
  std::transform(tmp_inputs_order.begin(), tmp_inputs_order.end(), std::back_inserter(new_subgraph_inputs),
                 [](std::pair<int, int> iter) { return iter.second; });
  subgraph_inputs.assign(new_subgraph_inputs.begin(), new_subgraph_inputs.end());

  return RET_OK;
}

STATUS SingleSwitchPass::UpdateSubgraphOutput(const size_t &subgraph_index, schema::CNodeT *partial_node,
                                              const std::vector<schema::CNodeT *> &subgraph_nodes) {
  if (partial_node == nullptr) {
    MS_LOG(ERROR) << "partial_node is nullptr or subgraph_nodes are empty.";
    return RET_INPUT_PARAM_INVALID;
  }
  auto &partial_outputs = partial_node->outputIndex;
  auto &subgraph = graph_->subGraph.at(subgraph_index);
  auto &subgraph_outputs = subgraph->outputIndices;

  std::map<int, int> subgraph_output_map;
  std::vector<std::pair<int, int>> tmp_outputs_order{};
  for (unsigned int &subgraph_output : subgraph_outputs) {
    for (auto &node : subgraph_nodes) {
      if (IsContain(node->outputIndex, subgraph_output)) {
        int partial_idx = GetSubgraphOutputTensorIndex(subgraph, node);
        if (partial_idx == -1) {
          MS_LOG(ERROR) << "get input index failed.";
          return RET_ERROR;
        }
        subgraph_output_map.insert(std::pair<int, int>{subgraph_output, partial_outputs[partial_idx]});
        tmp_outputs_order.emplace_back(partial_idx, partial_outputs[partial_idx]);
      }
    }
  }

  for (auto &subgraph_node : subgraph_nodes) {
    for (auto &output : subgraph_node->outputIndex) {
      if (subgraph_output_map.find(output) != subgraph_output_map.end()) {
        output = subgraph_output_map.at(output);
      }
    }
    for (auto &input : subgraph_node->inputIndex) {
      if (subgraph_output_map.find(input) != subgraph_output_map.end()) {
        input = subgraph_output_map.at(input);
      }
    }
  }

  std::vector<int> new_subgraph_outputs{};
  std::transform(tmp_outputs_order.begin(), tmp_outputs_order.end(), std::back_inserter(new_subgraph_outputs),
                 [](std::pair<int, int> iter) { return iter.second; });
  subgraph_outputs.assign(new_subgraph_outputs.begin(), new_subgraph_outputs.end());

  // filter for -1 output index
  std::vector<uint32_t> new_partial_outputs;
  std::copy_if(partial_outputs.begin(), partial_outputs.end(),
               std::inserter(new_partial_outputs, new_partial_outputs.begin()),
               [](uint32_t output) { return output != UINT32_MAX; });
  partial_node->outputIndex = new_partial_outputs;

  return RET_OK;
}

STATUS SingleSwitchPass::ConcatCondSubgraphInputAndOutput() {
  if (first_subgraph_index_ == -1) {
    MS_ASSERT(first_partial_node_->primitive != nullptr);
    MS_ASSERT(first_partial_node_->primitive->value..AsPartialFusion() != nullptr);
    first_partial_node_->primitive->value.AsPartialFusion()->sub_graph_index = -1;
    return RET_OK;
  }
  int ret = UpdateSubgraphInput(first_subgraph_index_, first_partial_node_, first_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }
  ret = UpdateSubgraphOutput(first_subgraph_index_, first_partial_node_, first_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }

  return ret;
}

STATUS SingleSwitchPass::ConcatBodySubgraphInputAndOutput() {
  if (second_subgraph_index_ == -1) {
    MS_ASSERT(first_partial_node_->primitive != nullptr);
    MS_ASSERT(first_partial_node_->primitive->value..AsPartialFusion() != nullptr);
    first_partial_node_->primitive->value.AsPartialFusion()->sub_graph_index = -1;
    return RET_OK;
  }
  int ret = UpdateSubgraphInput(second_subgraph_index_, second_partial_node_, second_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateSubgraphInput failed, ret: " << ret;
    return ret;
  }
  ret = UpdateSubgraphOutput(second_subgraph_index_, second_partial_node_, second_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateSubgraphOutput failed, ret: " << ret;
    return ret;
  }
  return ret;
}

STATUS SingleSwitchPass::Run() {
  int ret = Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }

  // switch converted from while
  if (IsLoop()) {
    ret = DoubleSwitchOutput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DoubleSwitchOutput failed, ret: " << ret;
      return ret;
    }

    ret = UpdateSwitchUser();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "UpdateOriginSwitchOutput failed, ret: " << ret;
      return ret;
    }

    ret = InsertMerge();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InsertMerge failed, ret: " << ret;
      return ret;
    }
  } else {  // switch converted from if
    ret = InsertPartialAndMergeAfterSwitch();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InsertPartialAndMergeAfterSwitch failed, ret: " << ret;
      return ret;
    }
  }

  ret = ConcatCondSubgraphInputAndOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConcatCondSubgraphInputAndOutput failed, ret: " << ret;
    return ret;
  }

  ret = ConcatBodySubgraphInputAndOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConcatBodySubgraphInputAndOutput failed, ret: " << ret;
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
