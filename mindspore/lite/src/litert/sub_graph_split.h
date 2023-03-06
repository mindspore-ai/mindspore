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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_SPLIT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_SPLIT_H_

#include <stack>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include "include/model.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/lite_model.h"
#include "src/litert/inner_context.h"
#include "src/common/prim_util.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
constexpr int kDefaultSubGraphSize = 2;
constexpr int kDefaultFirstSubgraph = 0;
constexpr int kDefaultSecondSubgraph = 1;
constexpr int kDefaultInputs = 1;
constexpr int kMaxMultyInNode = 20;
constexpr int kMaxSubGraphCount = 10;
constexpr int kMinSubgraphCost = 50;
constexpr double kDefaultGpu = 0.5;
class SearchSubGraph {
 public:
  enum TensorType { NORMAL, CONSTANT, INPUT };

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
    void empty() {
      io_cost_ = 0;
      mul_cost_ = 0;
    }
  };

  struct Subgraph {
    std::vector<uint32_t> nodes_;
    std::vector<uint32_t> heads_;
    std::vector<uint32_t> ends_;
    bool search_terminate_ = false;
    DeviceType device_;
    size_t thread_;
    CostModel cost_;
    uint32_t tid_; /* 1 or 2 */
  };

 public:
  SearchSubGraph(const InnerContext *context, Model *model, std::vector<lite::Tensor *> *src_tensors,
                 const std::map<int, OpParameter *> *op_parameters, std::vector<size_t> *output_nodes);
  ~SearchSubGraph() = default;

 public:
  void SubGraphSplit();
  void SubGraphSplitByOperator();
  void InsertNodeBegin(uint32_t index, Subgraph *subgraph, std::vector<size_t> *outputs);

 private: /* split by output */
  void SubGraphSplitByOutput();
  void InitSearchSubGraphByOutput();
  void InsertNode(uint32_t index, Subgraph *subgraph, uint32_t last_index);

 private: /* split by middle */
  void SubGraphSplitByMiddle();
  void InitSearchSubGraphByMiddle();
  void SearchMultyInNodes(std::vector<uint32_t> *multy_in_nodes);
  void InitMiddleSubgraph(const std::vector<uint32_t> *multy_in_nodes);
  void InsertNodeByMid(uint32_t node_index, Subgraph *subgraph, uint32_t last_index);
  void InsertHeadNode(uint32_t index, Subgraph *subgraph);
  void OptimizeAfterFusion(std::vector<Subgraph> *sub_graphs, uint32_t root_node_index);

 private: /* split by offline */
  void SubGraphSplitByOffLineParallel();
  void UpdateOfflineParallelFlag();
  bool CheckIsParallelSubGraph(const std::vector<Subgraph> &subgraphs);

 private: /* public graph func  */
  void RemoveConstNode(std::vector<uint32_t> *nodes);
  void InitSearchTensor();
  void InitMainGraphDevice(DeviceType dt = DT_CPU);

  void InitSubgraphRuntimeInfo(std::vector<Subgraph> *sub_graphs);
  void SubgraphFusion(std::vector<Subgraph> *sub_graphs);
  void CalculateCostModel(std::vector<Subgraph> *sub_graphs);
  void ConvertSubGraphToModel(std::vector<Subgraph> *sub_graphs);
  bool ValidInParallel();
  void CheckSubHeadEnd(Subgraph *sub);

 private: /* public schema func  */
  void InsertParallelNode(uint32_t index, Subgraph *subgraph);
  bool IsNodeSubGraphHead(uint32_t node_index, const std::vector<uint32_t> &ready_nodes);
  bool IsNodeSubGraphHeadWithRoot(uint32_t node_index, const std::vector<uint32_t> &ready_nodes,
                                  uint32_t root_node_index);
  const schema::Primitive *CreatePartialPrimitive(int64_t subgraph_index);

 private: /* public cost-model func  */
  CostModel CalculateConv2DFusion(const LiteGraph::Node *node);
  void dfs(int i, int n, int current_sum, int except_value, int *min_value, std::vector<bool> *tmp_group,
           std::vector<bool> *cor_group, std::vector<Subgraph> *sub_graphs);

 public:
  const InnerContext *context_ = nullptr;
  LiteModel *model_ = nullptr;
  std::vector<Tensor> tensors_;

 private:
  std::vector<size_t> *output_nodes_ = nullptr;
  std::vector<lite::Tensor *> *src_tensors_ = nullptr;
  const std::map<int, OpParameter *> *op_parameters_ = nullptr;
  std::vector<Subgraph> sub_graphs_;
  std::unordered_map<uint32_t, std::vector<Subgraph>> node_sub_map_;
  std::vector<LiteGraph::Node *> node_list_;
  DeviceType major_dt_;
  DeviceType minor_dt_;
  size_t major_thread_;
  size_t minor_thread_;
  size_t total_cost_ = 0;
  bool offline_parallel_enable_ = false;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_SPLIT_H_
