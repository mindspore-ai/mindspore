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

#ifndef MINDSPORE_LITE_SRC_SUB_GRAPH_SPLIT_H_
#define MINDSPORE_LITE_SRC_SUB_GRAPH_SPLIT_H_

#include <stack>
#include <vector>
#include "include/model.h"
#include "src/lite_kernel.h"
#include "src/lite_model.h"

namespace mindspore::lite {
#ifdef SUBGRAPH_SPLIT
class SearchSubGraph {
  enum TensorType { NORMAL, CONST, INPUT };

  struct Tensor {
    std::vector<uint32_t> in_nodes_; /* used current tensor as input */
    std::vector<uint32_t> out_nodes_;
    TensorType type_;
  };

  struct Subgraph {
    std::vector<uint32_t> nodes_;
    std::vector<uint32_t> heads_;
    std::vector<uint32_t> ends_;
    bool search_terminate_ = false;
    mindspore::kernel::KERNEL_ARCH device_;
  };

 public:
  SearchSubGraph(Model *model, std::vector<size_t> output_nodes) {
    output_nodes_.insert(output_nodes_.end(), output_nodes.begin(), output_nodes.end());
    node_list_ = model->all_nodes_;
    model_ = reinterpret_cast<LiteModel *>(model);
  }
  ~SearchSubGraph() = default;

 public:
  void SubGraphSplitByOutput();

 private:
  void InitSearchTensor();
  void InitSearchSubGraph();
  void ConvertSubGraphToModel();
  void InsertNode(uint32_t index, Subgraph *subgraph);
  bool IsNodeSubGraphHead(uint32_t node_index, const std::vector<uint32_t> &ready_nodes);
  const schema::Primitive *CreatePartialPrimitive(int64_t subgraph_index);
  void InitSubgraphDevice();
  void SubgraphFusion();
  void InitMainGraphDevice();

 private:
  LiteModel *model_ = nullptr;
  std::vector<Tensor> tensors_;
  std::vector<Subgraph> sub_graphs_;
  std::vector<size_t> output_nodes_;
  std::vector<Model::Node *> node_list_;
};

#endif
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SUB_GRAPH_SPLIT_H_
