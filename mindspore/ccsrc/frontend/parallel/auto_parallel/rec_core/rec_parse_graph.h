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

#ifndef PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
#define PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
static const std::set<OperatorType> ElementWiseOpType = {
  OperatorType::kRecReLU,         OperatorType::kRecLog,        OperatorType::kRecExp,
  OperatorType::kRecAdd,          OperatorType::kRecElmWiseOp,  OperatorType::kRecBiasAdd,
  OperatorType::kRecSub,          OperatorType::kRecMul,        OperatorType::kRecDiv,
  OperatorType::kRecSqueeze,      OperatorType::kRecReduce,     OperatorType::kRecCast,
  OperatorType::kRecReshape,      OperatorType::kRecGatherV2,   OperatorType::kRecArgWithValue,
  OperatorType::kRecSoftmax,      OperatorType::kRecOneHot,     OperatorType::kRecExpandDims,
  OperatorType::kRecStridedSlice, OperatorType::kRecBatchMatMul};

const TensorParam MakeTensor(int64_t n, int64_t c, int64_t h, int64_t w);

Graph::NodeType MakeNewOperator(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops);

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor);

TensorParam Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                             const size_t iter_input_tensors, Graph::NodeType NewTensor);

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names,
                                  const FuncGraphPtr &root);

void MakeEdge(const std::vector<std::vector<std::string>> &input_tensor_names, const std::shared_ptr<Graph> &graph);

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>> &input_tensor_name,
                                  const std::string &input_name);

void Eliminate_Aux_Outgoing(size_t node_index, const std::shared_ptr<Graph> &graph);

void Eliminate_Aux(size_t node_index, const std::shared_ptr<Graph> &graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list);

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> &graph,
                                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                      const std::shared_ptr<std::vector<size_t>> &index_list);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
