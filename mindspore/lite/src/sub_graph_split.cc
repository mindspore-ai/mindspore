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

#include "src/sub_graph_split.h"
#include <vector>
#include <utility>
#include "src/tensor.h"
#include "schema/inner/ops_generated.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {
#ifdef SUBGRAPH_SPLIT
const schema::Primitive *SearchSubGraph::CreatePartialPrimitive(int64_t subgraph_index) {
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto val_offset = schema::CreatePartialFusion(fbb, subgraph_index);
  auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_PartialFusion, val_offset.o);
  fbb.Finish(prim_offset);
  auto tmp_buf = fbb.GetBufferPointer();
  auto prim_buf = reinterpret_cast<char *>(malloc(fbb.GetSize()));
  memcpy(prim_buf, tmp_buf, fbb.GetSize());

  auto primitive = flatbuffers::GetRoot<schema::Primitive>(prim_buf);
  fbb.Clear();

  model_->node_bufs_.push_back(prim_buf);
  return std::move(primitive);
}

void SearchSubGraph::ConvertSubGraphToModel() {
  Model::SubGraph *main_graphs = model_->sub_graphs_.front();

  for (Subgraph &subgraph : sub_graphs_) {
    if (subgraph.nodes_.empty()) {
      continue;
    }
    mindspore::kernel::KERNEL_ARCH device = subgraph.device_;

    int new_sub_index = model_->sub_graphs_.size();
    int partial_index = model_->all_nodes_.size();

    Model::SubGraph *new_sub_graph = new (std::nothrow) Model::SubGraph();
    if (new_sub_graph == nullptr) {
      MS_LOG(ERROR) << "New sub graph failed!";
      return;
    }
    new_sub_graph->name_ = "Subgraph-split-" + std::to_string(new_sub_index);

    Model::Node *new_partial_node = new (std::nothrow) Model::Node();
    if (new_partial_node == nullptr) {
      MS_LOG(ERROR) << "New partial node failed!";
      return;
    }
    new_partial_node->name_ = "Partial-subgraph-split-" + std::to_string(new_sub_index);
    new_partial_node->node_type_ = mindspore::lite::NodeType_ValueNode;
    new_partial_node->primitive_ = CreatePartialPrimitive(new_sub_index);

    while (!subgraph.nodes_.empty()) {
      uint32_t node_index = subgraph.nodes_.front();
      new_sub_graph->node_indices_.push_back(node_index);
      VectorErase(&main_graphs->node_indices_, node_index);
      VectorErase(&subgraph.nodes_, node_index);
      model_->all_nodes_[node_index]->device_type_ = device;
    }

    for (uint32_t head_index : subgraph.heads_) {
      Model::Node *head_node = model_->all_nodes_[head_index];
      std::vector<uint32_t> inputs = head_node->input_indices_;
      for (auto input : inputs) {
        if (tensors_[input].type_ == CONST) {
          continue;
        }
        if (std::find(new_sub_graph->input_indices_.begin(), new_sub_graph->input_indices_.end(), input) !=
            new_sub_graph->input_indices_.end()) {
          continue;
        }
        new_sub_graph->input_indices_.insert(new_sub_graph->input_indices_.end(), input);
        new_partial_node->input_indices_.insert(new_partial_node->input_indices_.end(), input);
      }
    }

    for (uint32_t end_index : subgraph.ends_) {
      Model::Node *end_node = model_->all_nodes_[end_index];
      std::vector<uint32_t> outputs = end_node->output_indices_;
      new_sub_graph->output_indices_.insert(new_sub_graph->output_indices_.end(), outputs.begin(), outputs.end());
      new_partial_node->output_indices_.insert(new_partial_node->output_indices_.end(), outputs.begin(), outputs.end());
    }

    main_graphs->node_indices_.push_back(partial_index);
    model_->all_nodes_.push_back(std::move(new_partial_node));
    model_->sub_graphs_.push_back(std::move(new_sub_graph));
  }
  return;
}

bool SearchSubGraph::IsNodeSubGraphHead(uint32_t node_index, const std::vector<uint32_t> &ready_nodes) {
  std::vector<uint32_t> output_indexes = node_list_[node_index]->output_indices_;
  std::vector<uint32_t> output_nodes;
  for (uint32_t out_t : output_indexes) {
    std::vector<uint32_t> cur_nodes = tensors_[out_t].in_nodes_;
    output_nodes.insert(output_nodes.end(), cur_nodes.begin(), cur_nodes.end());
  }
  for (uint32_t out_n : output_nodes) {
    if (find(ready_nodes.begin(), ready_nodes.end(), out_n) == ready_nodes.end()) {
      return true;
    }
  }
  return false;
}

void SearchSubGraph::InsertNode(uint32_t index, Subgraph *subgraph) {
  if (subgraph->search_terminate_) {
    return;
  }

  Model::Node *node = node_list_[index];
  if (node == nullptr) {
    return;
  }

  std::vector<uint32_t> input = node->input_indices_;
  /* remove const node */
  for (int i = input.size() - 1; i >= 0; i--) {
    if (tensors_[input[i]].type_ == CONST) {
      input.erase(input.begin() + i);
    }
  }

  /* all node_input is graph_input */
  for (size_t i = 0; i < input.size(); i++) {
    if (tensors_[input[i]].type_ != INPUT) {
      break;
    }
    subgraph->heads_.clear();
    subgraph->ends_.clear();
    subgraph->nodes_.clear();
    subgraph->search_terminate_ = true;
    return;
  }

  /* split in graph */
  if (IsNodeSubGraphHead(index, subgraph->nodes_)) {
    if (subgraph->nodes_.empty()) {
      subgraph->search_terminate_ = true;
      return;
    }
    subgraph->heads_.push_back(subgraph->nodes_.front());
    return;
  }

  if (find(output_nodes_.begin(), output_nodes_.end(), index) != output_nodes_.end()) {
    subgraph->ends_.push_back(index);
  }

  /* node insert in current subgraph */
  subgraph->nodes_.insert(subgraph->nodes_.begin(), index);
  node_list_[index] = nullptr;

  /* search for next node */
  for (uint32_t in : input) {
    auto next_nodes = tensors_[in].out_nodes_;
    for (uint32_t next_node : next_nodes) {
      InsertNode(next_node, subgraph);
    }
  }
  return;
}

void SearchSubGraph::InitSearchSubGraph() {
  for (uint32_t out : output_nodes_) {
    Subgraph subgraph;

    InsertNode(out, &subgraph);

    sub_graphs_.push_back(std::move(subgraph));
  }
  return;
}

void SearchSubGraph::InitSearchTensor() {
  tensors_.resize(model_->all_tensors_.size());

  /* Set Tensor Type */
  for (size_t i = 0; i < tensors_.size(); i++) {
    tensors_[i].type_ = NORMAL;
    mindspore::schema::Tensor *src_tensor = model_->all_tensors_[i];
    auto category = TensorCategory(src_tensor);
    if (category == mindspore::lite::Tensor::Category::CONST_TENSOR ||
        category == mindspore::lite::Tensor::Category::CONST_SCALAR) {
      tensors_[i].type_ = CONST;
    }
  }
  std::vector<uint32_t> graph_input = model_->sub_graphs_[0]->input_indices_;
  for (auto in : graph_input) {
    tensors_[in].type_ = INPUT;
  }

  /* Set Tensor In and out Node */
  for (size_t index = 0; index < model_->all_nodes_.size(); index++) {
    Model::Node *node = model_->all_nodes_[index];
    std::vector<uint32_t> input = node->input_indices_;
    for (uint32_t in : input) {
      tensors_[in].in_nodes_.push_back(index);
    }
    std::vector<uint32_t> output = node->output_indices_;
    for (uint32_t out : output) {
      tensors_[out].out_nodes_.push_back(index);
    }
  }
  return;
}

void SearchSubGraph::InitSubgraphDevice() {
  sub_graphs_[0].device_ = kernel::KERNEL_ARCH::kCPU;
  sub_graphs_[1].device_ = kernel::KERNEL_ARCH::kALL;
}

void SearchSubGraph::InitMainGraphDevice() {
  kernel::KERNEL_ARCH main_device = kernel::KERNEL_ARCH::kALL;
  Model::SubGraph *main_graph = model_->sub_graphs_.front();
  for (uint32_t node_index : main_graph->node_indices_) {
    Model::Node *node = model_->all_nodes_[node_index];
    node->device_type_ = main_device;
  }
}

void SearchSubGraph::SubgraphFusion() {
  Subgraph new_npu_sub;
  Subgraph &npu_sub1 = sub_graphs_[1];
  Subgraph &npu_sub2 = sub_graphs_[2];
  new_npu_sub.nodes_.insert(new_npu_sub.nodes_.end(), npu_sub1.nodes_.begin(), npu_sub1.nodes_.end());
  new_npu_sub.nodes_.insert(new_npu_sub.nodes_.end(), npu_sub2.nodes_.begin(), npu_sub2.nodes_.end());
  new_npu_sub.heads_.insert(new_npu_sub.heads_.end(), npu_sub1.heads_.begin(), npu_sub1.heads_.end());
  new_npu_sub.heads_.insert(new_npu_sub.heads_.end(), npu_sub2.heads_.begin(), npu_sub2.heads_.end());
  new_npu_sub.ends_.insert(new_npu_sub.ends_.end(), npu_sub1.ends_.begin(), npu_sub1.ends_.end());
  new_npu_sub.ends_.insert(new_npu_sub.ends_.end(), npu_sub2.ends_.begin(), npu_sub2.ends_.end());
  sub_graphs_.erase(sub_graphs_.begin() + 2);
  sub_graphs_.erase(sub_graphs_.begin() + 1);
  sub_graphs_.insert(sub_graphs_.end(), std::move(new_npu_sub));
  return;
}

void SearchSubGraph::SubGraphSplitByOutput() {
  InitSearchTensor();

  InitSearchSubGraph();

  SubgraphFusion();

  InitSubgraphDevice();

  ConvertSubGraphToModel();

  InitMainGraphDevice();
}
#endif
}  // namespace mindspore::lite
