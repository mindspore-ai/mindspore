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
  MS_ASSERT(origin_switch_output_tensor_indices_.size() == cond_partial_node_->inputIndex.szie());
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

STATUS SingleSwitchPass::UpdateSwitchUser() {
  std::vector<CNodeT *> switch_users;
  for (auto &node_idx : graph_->subGraph.at(this_subgraph_index_)->nodeIndices) {
    auto &node = graph_->nodes.at(node_idx);
    for (auto &idx : node->inputIndex) {
      auto iter = std::find(switch_node_->outputIndex.begin(), switch_node_->outputIndex.end(), idx);
      if (iter != switch_node_->outputIndex.end()) {
        switch_users.push_back(node.get());
        int pos = iter - switch_node_->outputIndex.begin();
        idx = switch_node_->outputIndex.at(pos + switch_node_->outputIndex.size() / 2);
      }
    }
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

std::unique_ptr<schema::TensorT> SingleSwitchPass::NewTensor(const std::unique_ptr<schema::TensorT> &in_tensor) {
  auto out_tensor = std::make_unique<schema::TensorT>();
  out_tensor->nodeType = in_tensor->nodeType;
  out_tensor->dims = in_tensor->dims;
  out_tensor->dataType = in_tensor->dataType;
  out_tensor->data = in_tensor->data;
  out_tensor->format = in_tensor->format;
  return out_tensor;
}

STATUS SingleSwitchPass::MoveMaxIterationToCond() {
  auto &body_subgraph_input = graph_->subGraph.at(body_subgraph_index_)->inputIndices;
  for (auto it = body_subgraph_input.begin(); it != body_subgraph_input.end();) {
    if (!body_to_cond_partial_node_->inputIndex.empty() && IsContain(body_to_cond_partial_node_->inputIndex, *it)) {
      int32_t max_iteration_idx = it - body_subgraph_input.begin();
      // get maxiteration tensor
      auto &max_iteration_tensor = graph_->allTensors.at(cond_partial_node_->inputIndex.at(max_iteration_idx));
      auto all_tensor_idx = std::find(graph_->allTensors.begin(), graph_->allTensors.end(), max_iteration_tensor) -
                            graph_->allTensors.begin();

      // remove maxiteration from body_to_cond partial node
      body_to_cond_partial_node_->inputIndex.erase(body_to_cond_partial_node_->inputIndex.begin() + max_iteration_idx);

      // concat body subgraph tensor to max iteration in all tensor
      auto body_max_iteration_tensor_idx = body_subgraph_input.at(max_iteration_idx);
      for (auto &node : cond_graph_nodes_) {
        std::replace_if(
          node->inputIndex.begin(), node->inputIndex.end(),
          [&body_max_iteration_tensor_idx](uint32_t idx) { return idx == body_max_iteration_tensor_idx; },
          all_tensor_idx);
      }

      // remove maxiteration from body partial input and body func input
      body_partial_node_->inputIndex.erase(body_partial_node_->inputIndex.begin() + max_iteration_idx);
      it = body_subgraph_input.erase(it);
    } else {
      it++;
    }
  }
  return RET_OK;
}

STATUS SingleSwitchPass::InsertMerge() {
  int ret = RET_OK;
  auto merge_node = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
  MS_ASSERT(merge_node != nullptr);
  auto primitiveT = std::unique_ptr<PrimitiveT>(new (std::nothrow) PrimitiveT);
  MS_ASSERT(primitiveT != nullptr);
  merge_node->primitive = std::move(primitiveT);

  static int id = 0;
  merge_node->name = "Merge-" + std::to_string(id++);
  merge_node->primitive->value.type = schema::PrimitiveType_Merge;
  std::unique_ptr<MergeT> merge_param(new (std::nothrow) MergeT());
  MS_ASSERT(merge_param != nullptr);
  merge_node->primitive->value.value = merge_param.release();

  merge_node->inputIndex.assign(cond_partial_node_->inputIndex.begin(), cond_partial_node_->inputIndex.end());

  // merge node output is same as switch
  for (auto &out_index : origin_switch_output_tensor_indices_) {
    auto &switch_out_tensor = graph_->allTensors.at(out_index);
    auto tensor = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    merge_node->outputIndex.push_back(graph_->allTensors.size() - 1);
  }

  // double merge inputs to contain the outputs of body node
  for (auto &out_index : origin_switch_output_tensor_indices_) {
    auto &switch_out_tensor = graph_->allTensors.at(out_index);
    auto tensor = NewTensor(switch_out_tensor);
    graph_->allTensors.push_back(std::move(tensor));
    merge_node->inputIndex.push_back(graph_->allTensors.size() - 1);
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

  // update bodu graph output
  graph_->subGraph.at(body_subgraph_index_)
    ->outputIndices.assign(body_to_cond_partial_node_->inputIndex.begin(),
                           body_to_cond_partial_node_->inputIndex.end());

  // erase body_to_cond_partial_node_
  RemoveUselessNode(body_to_cond_partial_node_, graph_);
  return ret;
}

void SingleSwitchPass::RemoveUselessNode(schema::CNodeT *partial_node, schema::MetaGraphT *graph) {
  partial_node->inputIndex.clear();
  partial_node->outputIndex.clear();

  int pos = -1;
  for (size_t i = 0; i < graph->nodes.size(); ++i) {
    if (graph->nodes.at(i).get() == partial_node) {
      pos = i;
      break;
    }
  }

  if (pos == -1) {
    return;
  }

  graph->nodes.erase(graph->nodes.begin() + pos);

  for (auto &subgraph : graph->subGraph) {
    for (auto it = subgraph->nodeIndices.begin(); it != subgraph->nodeIndices.end();) {
      if (*it == static_cast<uint32_t>(pos)) {
        it = subgraph->nodeIndices.erase(it);
      } else {
        if (*it > static_cast<uint32_t>(pos)) {
          (*it)--;
        }
        it++;
      }
    }
  }
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

  if (switch_node_->inputIndex.size() == kSwitchMinInputSize) {
    return RET_OK;
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
        cond_node_index_ = iter - graph_->nodes.begin();
        find_cond_node = true;
      }
      if (out_index == switch_node_->inputIndex[kSwitchBodyIndex]) {
        body_partial_node_ = iter->get();
        body_node_index_ = iter - graph_->nodes.begin();
        find_body_node = true;
      }
    }
    if (find_body_node && find_cond_node) {
      break;
    }
  }

  if (cond_partial_node_->primitive->value.type != PrimitiveType_Partial ||
      body_partial_node_->primitive->value.type != PrimitiveType_Partial) {
    return RET_OK;
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

STATUS SingleSwitchPass::UpdateSubgraphInput(const size_t &subgraph_index, schema::CNodeT *partial_node,
                                             const std::vector<schema::CNodeT *> &subgraph_nodes) {
  if (partial_node == nullptr || subgraph_nodes.empty()) {
    MS_LOG(ERROR) << "partial_node is nullptr or subgraph_nodes are empty.";
    return RET_INPUT_PARAM_INVALID;
  }
  auto &partial_inputs = partial_node->inputIndex;
  auto &subgraph_inputs = graph_->subGraph.at(subgraph_index)->inputIndices;

  std::map<int, int> subgraph_input_map;
  std::vector<int> new_subgraph_inputs{};
  for (unsigned int &subgraph_input : subgraph_inputs) {
    auto &tensor = graph_->allTensors.at(subgraph_input);
    // get parameter input index k. subgraph name + “_input_" + "k"
    char k = tensor->name[graph_->subGraph.at(subgraph_index)->name.size() + 7];
    int partial_idx = k - '0';
    subgraph_input_map.insert(std::pair<int, int>{subgraph_input, partial_inputs[partial_idx]});
    new_subgraph_inputs.push_back(partial_inputs[partial_idx]);
  }

  for (auto &subgraph_node : subgraph_nodes) {
    for (auto &input : subgraph_node->inputIndex) {
      if (subgraph_input_map.find(input) != subgraph_input_map.end()) {
        input = subgraph_input_map.at(input);
      }
    }
  }
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
  auto &subgraph_outputs = graph_->subGraph.at(subgraph_index)->outputIndices;

  std::map<int, int> subgraph_output_map;
  std::vector<int> new_subgraph_outputs{};
  for (unsigned int &subgraph_output : subgraph_outputs) {
    auto &tensor = graph_->allTensors.at(subgraph_output);
    // get parameter input index k. subgraph name + “_output_" + "k"
    char k = tensor->name[graph_->subGraph.at(subgraph_index)->name.size() + 8];
    int partial_idx = k - '0';
    subgraph_output_map.insert(std::pair<int, int>{subgraph_output, partial_outputs[partial_idx]});
    new_subgraph_outputs.push_back(partial_outputs[partial_idx]);
  }

  for (auto &subgraph_node : subgraph_nodes) {
    for (auto &output : subgraph_node->outputIndex) {
      if (subgraph_output_map.find(output) != subgraph_output_map.end()) {
        output = subgraph_output_map.at(output);
      }
    }
  }
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

STATUS SingleSwitchPass::ConvertSwitchToSelect() {
  MS_ASSERT(switch_node_->inputIndex.size() >= 3);
  MS_ASSERT(switch_node_->inputIndex.size() % 2 != 0);
  MS_ASSERT(switch_node_->outputIndex.size() * 2 + 1 == switch_node_->inputIndex.size());
  auto bool_index = switch_node_->inputIndex.front();

  // insert switch node1
  auto switch_node1 = std::make_unique<CNodeT>();
  switch_node1->name = switch_node_->name + "-Switch-1";
  switch_node1->primitive = std::make_unique<PrimitiveT>();
  switch_node1->primitive->value.type = PrimitiveType_Switch;
  switch_node1->primitive->value.value = new (std::nothrow) SwitchT();
  switch_node1->inputIndex = {bool_index};
  std::vector<int> part_one_input_index(
    switch_node_->inputIndex.begin() + 1,
    switch_node_->inputIndex.begin() + 1 + (switch_node_->inputIndex.size() - 1) / 2);
  switch_node1->inputIndex.insert(switch_node1->inputIndex.end(), part_one_input_index.begin(),
                                  part_one_input_index.end());
  std::vector<std::unique_ptr<TensorT>> switch_output_tensors1(part_one_input_index.size() * 2);
  std::vector<int> switch_output_indexes1(part_one_input_index.size() * 2);
  int i = 0;
  for (const auto &input_index : part_one_input_index) {
    auto &switch_in_tensor = graph_->allTensors.at(input_index);
    auto tensor1 = NewTensor(switch_in_tensor);
    auto tensor2 = NewTensor(switch_in_tensor);
    switch_output_tensors1[i] = std::move(tensor1);
    switch_output_tensors1[part_one_input_index.size() + i] = std::move(tensor2);
    switch_output_indexes1[i] = graph_->allTensors.size() - 1 + i;
    switch_output_indexes1[part_one_input_index.size() + i] =
      graph_->allTensors.size() - 1 + i + part_one_input_index.size();
    i++;
  }
  for (auto &tensor : switch_output_tensors1) {
    graph_->allTensors.emplace_back(std::move(tensor));
  }
  switch_node1->outputIndex.insert(switch_node1->outputIndex.begin(), switch_output_indexes1.begin(),
                                   switch_output_indexes1.end());

  // insert switch node2
  auto switch_node2 = std::make_unique<CNodeT>();
  switch_node2->name = switch_node_->name + "-Switch-1";
  switch_node2->primitive = std::make_unique<PrimitiveT>();
  switch_node2->primitive->value.type = PrimitiveType_Switch;
  switch_node2->primitive->value.value = new (std::nothrow) SwitchT();
  switch_node2->inputIndex = {bool_index};

  std::vector<int> part_two_input_index(
    switch_node_->inputIndex.begin() + 1 + (switch_node_->inputIndex.size() - 1) / 2, switch_node_->inputIndex.end());
  switch_node2->inputIndex.insert(switch_node2->inputIndex.end(), part_two_input_index.begin(),
                                  part_two_input_index.end());
  std::vector<std::unique_ptr<TensorT>> switch_output_tensors2(part_two_input_index.size() * 2);
  std::vector<int> switch_output_indexes2(part_two_input_index.size() * 2);
  i = 0;
  for (const auto &input_index : part_two_input_index) {
    auto &switch_in_tensor = graph_->allTensors.at(input_index);
    auto tensor1 = NewTensor(switch_in_tensor);
    auto tensor2 = NewTensor(switch_in_tensor);
    switch_output_tensors2[i] = std::move(tensor1);
    switch_output_tensors2[part_two_input_index.size() + i] = std::move(tensor2);
    switch_output_indexes2[i] = graph_->allTensors.size() - 1 + i;
    switch_output_indexes2[part_two_input_index.size() + i] =
      graph_->allTensors.size() - 1 + i + part_two_input_index.size();
    i++;
  }
  for (auto &tensor : switch_output_tensors2) {
    graph_->allTensors.emplace_back(std::move(tensor));
  }
  switch_node2->outputIndex.insert(switch_node2->outputIndex.begin(), switch_output_indexes2.begin(),
                                   switch_output_indexes2.end());

  // insert merge
  auto merge_node = std::make_unique<CNodeT>();
  merge_node->name = switch_node_->name + "-Merge";
  merge_node->primitive = std::make_unique<PrimitiveT>();
  merge_node->primitive->value.type = PrimitiveType_Merge;
  merge_node->primitive->value.value = new (std::nothrow) MergeT();

  std::vector<int> merge_input_indexes(switch_node_->outputIndex.size() * 2);
  for (i = 0; i < switch_node_->outputIndex.size(); i++) {
    merge_input_indexes[i] = switch_output_indexes1[i];
    merge_input_indexes[i + switch_node_->outputIndex.size()] =
      switch_output_indexes2[i + switch_node_->outputIndex.size()];
    merge_node->outputIndex.emplace_back(switch_node_->outputIndex.at(i));
  }
  merge_node->inputIndex.insert(merge_node->inputIndex.end(), merge_input_indexes.begin(), merge_input_indexes.end());
  graph_->nodes.emplace_back(std::move(switch_node1));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.push_back(graph_->nodes.size() - 1);
  graph_->nodes.emplace_back(std::move(switch_node2));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.push_back(graph_->nodes.size() - 1);
  graph_->nodes.emplace_back(std::move(merge_node));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.push_back(graph_->nodes.size() - 1);

  RemoveUselessNode(switch_node_, graph_);
  return RET_OK;
}

STATUS SingleSwitchPass::Run() {
  int ret = Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }

  if (switch_node_->inputIndex.size() == kSwitchMinInputSize) {
    return RET_OK;
  }

  if (cond_partial_node_->primitive->value.type != PrimitiveType_Partial ||
      body_partial_node_->primitive->value.type != PrimitiveType_Partial) {
    ret = ConvertSwitchToSelect();
    return ret;
  }

  if (IsLoop()) {
    ret = MoveMaxIterationToCond();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MoveMaxIterationToCond failed, ret: " << ret;
      return ret;
    }
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
