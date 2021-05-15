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
#include <map>
#include "include/model.h"
#include "src/lite_kernel.h"
#include "src/lite_model.h"
#include "src/inner_context.h"
#include "src/common/prim_util.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
class SearchSubGraph {
  enum TensorType { NORMAL, CONST, INPUT };

  struct Tensor {
    std::vector<uint32_t> in_nodes_; /* used current tensor as input */
    std::vector<uint32_t> out_nodes_;
    TensorType type_;
  };

  struct CostModel {
    size_t mul_cost_ = 0;
    size_t io_cost_ = 0;

    CostModel operator+(const SearchSubGraph::CostModel &cost) {
      CostModel result;
      result.mul_cost_ = this->mul_cost_ + cost.mul_cost_;
      result.io_cost_ = this->io_cost_ + cost.io_cost_;
      return result;
    }
    CostModel operator-(const SearchSubGraph::CostModel &cost) {
      CostModel result;
      result.mul_cost_ = this->mul_cost_ - cost.mul_cost_;
      result.io_cost_ = this->io_cost_ - cost.io_cost_;
      return result;
    }
    int cost() { return io_cost_ + mul_cost_; }
  };

  struct Subgraph {
    std::vector<uint32_t> nodes_;
    std::vector<uint32_t> heads_;
    std::vector<uint32_t> ends_;
    bool search_terminate_ = false;
    DeviceType device_;
    CostModel cost_;
  };

 public:
  SearchSubGraph(const InnerContext *context, Model *model, std::vector<lite::Tensor *> *src_tensors,
                 const std::map<int, OpParameter *> *op_parameters, std::vector<size_t> output_nodes)
      : context_(context), src_tensors_(src_tensors), op_parameters_(op_parameters) {
    output_nodes_.insert(output_nodes_.end(), output_nodes.begin(), output_nodes.end());
    node_list_ = model->all_nodes_;
    model_ = reinterpret_cast<LiteModel *>(model);
    major_dt_ = DT_CPU;
    minor_dt_ = DT_CPU;
    if (context_->IsNpuEnabled()) {
      major_dt_ = DT_NPU;
    } else if (context_->IsGpuEnabled()) {
      major_dt_ = DT_GPU;
    }
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
  void CalculateCostModel();
  CostModel CalculateConv2DFusion(Model::Node *node);
  void dfs(int i, int n, int current_sum, int except_value, int *min_value, std::vector<bool> *tmp_group,
           std::vector<bool> *cor_group);

 private:
  const InnerContext *context_ = nullptr;
  LiteModel *model_ = nullptr;
  std::vector<lite::Tensor *> *src_tensors_ = nullptr;
  const std::map<int, OpParameter *> *op_parameters_ = nullptr;
  std::vector<Tensor> tensors_;
  std::vector<Subgraph> sub_graphs_;
  std::vector<size_t> output_nodes_;
  std::vector<Model::Node *> node_list_;
  DeviceType major_dt_;
  DeviceType minor_dt_;
  size_t total_cost_ = 0;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SUB_GRAPH_SPLIT_H_
