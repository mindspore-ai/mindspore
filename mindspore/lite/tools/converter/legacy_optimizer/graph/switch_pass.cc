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

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include "tools/converter/legacy_optimizer/graph/switch_pass.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/ops/primitive_c.h"
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
  return RET_OK;
}

STATUS SingleSwitchPass::DoubleSwitchOutput() {
  origin_switch_output_tensor_indices_ = switch_node_->outputIndex;
  if (origin_switch_output_tensor_indices_.size() != cond_partial_node_->inputIndex.size()) {
    MS_LOG(ERROR) << "switch node: " << switch_node_->name << " input or output number is not right.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < origin_switch_output_tensor_indices_.size(); i++) {
    auto &switch_out_tensor = graph_->allTensors.at(origin_switch_output_tensor_indices_[i]);
    const auto &cond_partial_input_tensor = graph_->allTensors.at(cond_partial_node_->inputIndex[i]);
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
  for (auto &node : body_graph_nodes_) {
    if (node->primitive->value.type == schema::PrimitiveType_Partial &&
        node->primitive->value.AsPartial()->subGraphIndex == cond_subgraph_index_) {
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
  auto &body_fg = graph_->subGraph.at(body_subgraph_index_);
  auto body_fg_output = body_fg->outputIndices;
  for (auto &subgraph_output : body_fg_output) {
    for (auto &node : body_graph_nodes_) {
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

STATUS SingleSwitchPass::InsertMerge() {
  // update body graph output
  auto &body_fg = graph_->subGraph.at(body_subgraph_index_);
  body_fg->outputIndices.assign(body_to_cond_partial_node_->inputIndex.begin(),
                                body_to_cond_partial_node_->inputIndex.end());

  // remove body_to_cond_partial_node_ from body_graph_nodes_
  for (auto it = body_graph_nodes_.begin(); it != body_graph_nodes_.end();) {
    if (*it == body_to_cond_partial_node_) {
      it = body_graph_nodes_.erase(it);
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
  for (size_t i = 0; i < body_partial_node_->inputIndex.size(); i++) {
    if (IsContain(variable_input, i)) {
      continue;
    }
    const_input.push_back(i);
  }

  auto merge_node = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
  auto primitiveT = std::unique_ptr<PrimitiveT>(new (std::nothrow) PrimitiveT);
  MS_ASSERT(primitiveT != nullptr);
  merge_node->primitive = std::move(primitiveT);

  static int id = 0;
  merge_node->name = "Merge-" + std::to_string(id++);
  merge_node->primitive->value.type = schema::PrimitiveType_Merge;
  std::unique_ptr<MergeT> merge_param(new (std::nothrow) MergeT());
  MS_ASSERT(merge_param != nullptr);
  merge_node->primitive->value.value = merge_param.release();

  // merge node output is same as switch
  for (auto &out_index : origin_switch_output_tensor_indices_) {
    auto &switch_out_tensor = graph_->allTensors.at(out_index);
    auto tensor = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    merge_node->outputIndex.push_back(graph_->allTensors.size() - 1);
  }

  merge_node->inputIndex.assign(cond_partial_node_->inputIndex.begin(), cond_partial_node_->inputIndex.end());

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
      tensor->nodeType = schema::NodeType_CNode;
      graph_->allTensors.push_back(std::move(tensor));
      merge_node->inputIndex.push_back(graph_->allTensors.size() - 1);
    }
  }

  // insert merge node before the cond graph
  std::map<int, int> cond_input_update_map{};
  for (size_t i = 0; i < cond_partial_node_->inputIndex.size(); i++) {
    cond_input_update_map.insert(std::make_pair(cond_partial_node_->inputIndex.at(i), merge_node->outputIndex.at(i)));
  }
  for (auto &node : cond_graph_nodes_) {
    for (auto &input_idx : node->inputIndex) {
      if (cond_input_update_map.find(input_idx) != cond_input_update_map.end()) {
        input_idx = cond_input_update_map.at(input_idx);
      }
    }
  }

  // update cond node input to be consistent with cond graph input
  cond_partial_node_->inputIndex.assign(merge_node->outputIndex.begin(), merge_node->outputIndex.end());

  // insert switch after cond node and merge node
  auto cond_input = switch_node_->inputIndex.front();
  switch_node_->inputIndex.clear();
  switch_node_->inputIndex.push_back(cond_input);
  switch_node_->inputIndex.insert(switch_node_->inputIndex.end(), merge_node->outputIndex.begin(),
                                  merge_node->outputIndex.end());

  // move body node to switch node output
  body_partial_node_->inputIndex.clear();
  body_partial_node_->inputIndex.assign(switch_node_->outputIndex.begin(),
                                        switch_node_->outputIndex.begin() + switch_node_->outputIndex.size() / 2);

  // concat body output to merge input
  body_partial_node_->outputIndex.assign(merge_node->inputIndex.begin() + merge_node->inputIndex.size() / 2,
                                         merge_node->inputIndex.end());

  graph_->nodes.push_back(std::move(merge_node));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.push_back(graph_->nodes.size() - 1);

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

  // get cond_partial_node_ and body_partial_node_
  bool find_cond_node = false;
  bool find_body_node = false;
  for (auto iter = graph_->nodes.begin(); iter < graph_->nodes.end(); iter++) {
    for (auto &out_index : iter->get()->outputIndex) {
      if (out_index == switch_node_->inputIndex[kSwitchCondIndex]) {
        cond_partial_node_ = iter->get();
        find_cond_node = true;
      }
      if (out_index == switch_node_->inputIndex[kSwitchBodyIndex]) {
        body_partial_node_ = iter->get();
        find_body_node = true;
      }
    }
    if (find_body_node && find_cond_node) {
      break;
    }
  }

  // get cond_graph_nodes_
  cond_subgraph_index_ = cond_partial_node_->primitive->value.AsPartial()->subGraphIndex;
  auto cond_node_indices = graph_->subGraph.at(cond_subgraph_index_)->nodeIndices;
  for (auto &index : cond_node_indices) {
    cond_graph_nodes_.push_back(graph_->nodes.at(index).get());
  }

  // get body_graph_nodes_
  body_subgraph_index_ = body_partial_node_->primitive->value.AsPartial()->subGraphIndex;
  auto body_node_indices = graph_->subGraph.at(body_subgraph_index_)->nodeIndices;
  for (auto &index : body_node_indices) {
    body_graph_nodes_.push_back(graph_->nodes.at(index).get());
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
  if (partial_node == nullptr || subgraph_nodes.empty()) {
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
  if (partial_node == nullptr || subgraph_nodes.empty()) {
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
  }

  std::vector<int> new_subgraph_outputs{};
  std::transform(tmp_outputs_order.begin(), tmp_outputs_order.end(), std::back_inserter(new_subgraph_outputs),
                 [](std::pair<int, int> iter) { return iter.second; });
  subgraph_outputs.assign(new_subgraph_outputs.begin(), new_subgraph_outputs.end());

  return RET_OK;
}

STATUS SingleSwitchPass::ConcatCondSubgraphInputAndOutput() {
  int ret = UpdateSubgraphInput(cond_subgraph_index_, cond_partial_node_, cond_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }
  ret = UpdateSubgraphOutput(cond_subgraph_index_, cond_partial_node_, cond_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }

  return ret;
}

STATUS SingleSwitchPass::ConcatBodySubgraphInputAndOutput() {
  int ret = UpdateSubgraphInput(body_subgraph_index_, body_partial_node_, body_graph_nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateSubgraphInput failed, ret: " << ret;
    return ret;
  }
  ret = UpdateSubgraphOutput(body_subgraph_index_, body_partial_node_, body_graph_nodes_);
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

  if (IsLoop()) {
    ret = InsertMerge();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InsertMerge failed, ret: " << ret;
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
