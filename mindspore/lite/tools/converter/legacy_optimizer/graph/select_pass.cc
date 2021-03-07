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

#include "tools/converter/legacy_optimizer/graph/select_pass.h"
#include <vector>
#include <map>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"

namespace mindspore::lite {
STATUS SelectPass::Run(mindspore::schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    auto type = node->primitive->value.type;
    if (type != schema::PrimitiveType_Select) {
      continue;
    }

    SingleSelectPass pass(graph, i);
    int ret = pass.Run();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "node: " << node->name << "'s select pass failed: " << ret;
      return ret;
    }
    select_indices_.emplace_back(i);
  }
  int ret = RemoveSelectNodes();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "remove select nodes failed";
    return ret;
  }
  return RET_OK;
}

STATUS SelectPass::RemoveSelectNodes() {
  std::sort(select_indices_.begin(), select_indices_.end(), std::greater<int>());
  for (auto select_indice : select_indices_) {
    auto &node = graph_->nodes.at(select_indice);
    if (node->primitive->value.type != PrimitiveType_Select) {
      MS_LOG(ERROR) << "node " << node->name << " is not a select node";
      return RET_ERROR;
    }
    int subgraph_idx = -1;
    for (size_t i = 0; i < graph_->subGraph.size(); i++) {
      if (IsContain(graph_->subGraph.at(i)->nodeIndices, select_indice)) {
        subgraph_idx = i;
        break;
      }
    }

    if (subgraph_idx == -1) {
      MS_LOG(ERROR) << "select node " << node->name << " is not belong to any subgraph";
      return RET_ERROR;
    }
    graph_->nodes.erase(graph_->nodes.begin() + select_indice);
    std::vector<uint32_t> new_node_indices;
    std::copy_if(graph_->subGraph.at(subgraph_idx)->nodeIndices.begin(),
                 graph_->subGraph.at(subgraph_idx)->nodeIndices.end(),
                 std::inserter(new_node_indices, new_node_indices.begin()),
                 [&select_indice](int indice) { return (uint32_t)indice != select_indice; });
    graph_->subGraph.at(subgraph_idx)->nodeIndices = new_node_indices;
    for (auto &subgraph : graph_->subGraph) {
      std::transform(subgraph->nodeIndices.begin(), subgraph->nodeIndices.end(), subgraph->nodeIndices.begin(),
                     [&select_indice](uint32_t idx) {
                       if (idx > select_indice) {
                         return --idx;
                       }
                       return idx;
                     });
    }
  }
  return RET_OK;
}

std::unique_ptr<schema::TensorT> SingleSelectPass::NewTensor(const std::unique_ptr<schema::TensorT> &in_tensor) {
  auto out_tensor = std::make_unique<schema::TensorT>();
  out_tensor->nodeType = in_tensor->nodeType;
  out_tensor->dims = in_tensor->dims;
  out_tensor->dataType = in_tensor->dataType;
  out_tensor->data = in_tensor->data;
  out_tensor->format = in_tensor->format;
  return out_tensor;
}

void SingleSelectPass::RemoveUselessNode(schema::CNodeT *partial_node) {
  partial_node->inputIndex.clear();
  partial_node->outputIndex.clear();
}

size_t SingleSelectPass::InitThisGraphIndex() {
  for (size_t i = 0; i < graph_->subGraph.size(); i++) {
    if (std::any_of(graph_->subGraph.at(i)->nodeIndices.begin(), graph_->subGraph.at(i)->nodeIndices.end(),
                    [this](const uint32_t &idx) { return idx == this->select_node_index_; })) {
      return i;
    }
  }
  return -1;
}

STATUS SingleSelectPass::Init() {
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return RET_NULL_PTR;
  }

  this_subgraph_index_ = InitThisGraphIndex();
  if (this_subgraph_index_ < 0) {
    MS_LOG(ERROR) << "init this subgraph index failed.";
    return RET_ERROR;
  }

  select_node_ = graph_->nodes.at(select_node_index_).get();
  if (select_node_ == nullptr) {
    MS_LOG(ERROR) << "select node is nullptr.";
    return RET_NULL_PTR;
  }

  if (select_node_->inputIndex.size() == kSelectMinInputSize &&
      select_node_->outputIndex.size() == kSelectMinOutputSize) {
    return RET_OK;
  }

  if (select_node_->inputIndex.size() < kSelectMinInputSize) {
    MS_LOG(ERROR) << "select node: " << select_node_->name
                  << " 's input size is not right, size: " << select_node_->inputIndex.size();
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

STATUS SingleSelectPass::ConvertSelectToSwitch() {
  MS_ASSERT(select_node_->inputIndex.size() >= 3);
  MS_ASSERT(select_node_->inputIndex.size() % 2 != 0);
  MS_ASSERT(select_node_->outputIndex.size() * 2 + 1 == select_node_->inputIndex.size());
  auto bool_index = select_node_->inputIndex.front();

  // insert switch node1
  auto switch_node1 = std::make_unique<CNodeT>();
  switch_node1->name = select_node_->name + "-Switch-1";
  switch_node1->primitive = std::make_unique<PrimitiveT>();
  switch_node1->primitive->value.type = PrimitiveType_Switch;
  switch_node1->primitive->value.value = new (std::nothrow) SwitchT();
  switch_node1->inputIndex = {bool_index};
  std::vector<int> part_one_input_index(
    select_node_->inputIndex.begin() + 1,
    select_node_->inputIndex.begin() + 1 + (select_node_->inputIndex.size() - 1) / 2);
  switch_node1->inputIndex.insert(switch_node1->inputIndex.end(), part_one_input_index.begin(),
                                  part_one_input_index.end());
  std::vector<std::unique_ptr<TensorT>> switch_output_tensors1(part_one_input_index.size() * 2);
  std::vector<int> switch_output_indexes1(part_one_input_index.size() * 2);
  size_t i = 0;
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
  switch_node2->name = select_node_->name + "-Switch-2";
  switch_node2->primitive = std::make_unique<PrimitiveT>();
  switch_node2->primitive->value.type = PrimitiveType_Switch;
  switch_node2->primitive->value.value = new (std::nothrow) SwitchT();
  switch_node2->inputIndex = {bool_index};

  std::vector<int> part_two_input_index(
    select_node_->inputIndex.begin() + 1 + (select_node_->inputIndex.size() - 1) / 2, select_node_->inputIndex.end());
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
  merge_node->name = select_node_->name + "-merge";
  merge_node->primitive = std::make_unique<PrimitiveT>();
  merge_node->primitive->value.type = PrimitiveType_Merge;
  merge_node->primitive->value.value = new (std::nothrow) MergeT();

  std::vector<int> merge_input_indexes(select_node_->outputIndex.size() * 2);
  for (i = 0; i < select_node_->outputIndex.size(); i++) {
    merge_input_indexes[i] = switch_output_indexes1[i];
    merge_input_indexes[i + select_node_->outputIndex.size()] =
      switch_output_indexes2[i + select_node_->outputIndex.size()];
    merge_node->outputIndex.emplace_back(select_node_->outputIndex.at(i));
  }
  merge_node->inputIndex.insert(merge_node->inputIndex.end(), merge_input_indexes.begin(), merge_input_indexes.end());
  graph_->nodes.emplace_back(std::move(switch_node1));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.emplace_back(graph_->nodes.size() - 1);
  graph_->nodes.emplace_back(std::move(switch_node2));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.emplace_back(graph_->nodes.size() - 1);
  graph_->nodes.emplace_back(std::move(merge_node));
  graph_->subGraph.at(this_subgraph_index_)->nodeIndices.emplace_back(graph_->nodes.size() - 1);

  RemoveUselessNode(select_node_);
  return RET_OK;
}

STATUS SingleSelectPass::Run() {
  int ret = Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init failed, ret: " << ret;
    return ret;
  }

  ret = ConvertSelectToSwitch();
  return ret;
}
}  // namespace mindspore::lite
