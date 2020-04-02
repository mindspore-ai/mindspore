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

#ifndef PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
#define PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "parallel/auto_parallel/rec_core/rec_graph.h"
#include "parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
const std::map<std::string, OperatorType> DictOpType{
  {MATMUL, OperatorType::kRecMatMul},
  {CONV2D, OperatorType::kRecConvolution},
  {MAXPOOLV2, OperatorType::kRecPooling},
  {SIMPLE_MEAN, OperatorType::kRecPooling},
  {TENSOR_ADD, OperatorType::kRecAdd},
  {RESHAPE, OperatorType::kRecReshape},
  {BIAS_ADD, OperatorType::kRecBiasAdd},
  {RELU, OperatorType::kRecReLU},
  {BATCH_NORM, OperatorType::kRecBatchNorm},
  {SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits},
};

const TensorParam MakeTensor(int n, int c, int h, int w);

bool IsInList(const std::string& name, const std::vector<std::string>& list);

Graph::NodeType MakeNewOperator(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops);

Graph::NodeType MakeNewTensor(std::vector<std::shared_ptr<OperatorInfo>> ops, const size_t iter_ops,
                              const std::string& input, const size_t iter_input_tensors, std::shared_ptr<Graph> graph,
                              size_t current_op_index);
void Fill2DTensor(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                  const std::shared_ptr<Graph> graph, const size_t iter_input_tensors, const size_t current_op_index,
                  Graph::NodeType NewTensor);
void CompleteOperatorInputs(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops, size_t iter_input_tensors,
                            size_t current_op_index, std::shared_ptr<Graph> graph);
void Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                      const std::shared_ptr<Graph> graph, const size_t iter_input_tensors,
                      const size_t current_op_index);
void MakeEdge(std::shared_ptr<Graph> graph, const size_t input_index, const size_t current_op_index);

void ModifyTensorToOperator(std::shared_ptr<Graph> graph, const size_t current_op_index, const size_t iter_ops,
                            std::vector<std::shared_ptr<OperatorInfo>> ops);

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>>& ops,
                                  const std::vector<std::vector<std::string>>& input_tensor_names,
                                  const std::shared_ptr<std::vector<size_t>>& ops_nodes_list);

void LinkOps(std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<OperatorInfo>> ops,
             const std::vector<std::vector<std::string>>& input_tensor_names, std::vector<std::string> current_graph,
             const size_t iter_ops, const size_t current_op_index);

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> graph,
                                      std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                                      std::shared_ptr<std::vector<size_t>> index_list);
void Eliminate_Aux(const size_t node_index, std::shared_ptr<Graph> graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
