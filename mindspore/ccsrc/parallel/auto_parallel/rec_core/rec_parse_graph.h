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
  {MAXPOOL, OperatorType::kRecPooling},
  {MAXPOOLV2, OperatorType::kRecPooling},
  {SIMPLE_MEAN, OperatorType::kRecPooling},
  {TENSOR_ADD, OperatorType::kRecElmWiseOp},
  {RESHAPE, OperatorType::kRecReshape},
  {BIAS_ADD, OperatorType::kRecBiasAdd},
  {RELU, OperatorType::kRecReLU},
  {BATCH_NORM, OperatorType::kRecBatchNorm},
  {FUSE_BATCH_NORM, OperatorType::kRecBatchNorm},
  {SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits},
  {SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits},
  {ONEHOT, OperatorType::kRecOneHot},
  {LOG, OperatorType::kRecLog},
  {EXP, OperatorType::kRecExp},
  {SUB, OperatorType::kRecElmWiseOp},
  {MUL, OperatorType::kRecElmWiseOp},
  {DIV, OperatorType::kRecElmWiseOp},
  {SQUEEZE, OperatorType::kRecSqueeze},
  {CAST, OperatorType::kRecCast},
  {REDUCE_SUM, OperatorType::kRecCast},
  {REDUCE_MAX, OperatorType::kRecCast},
  {REDUCE_MIN, OperatorType::kRecCast},
  {REDUCE_MEAN, OperatorType::kRecCast}};

const TensorParam MakeTensor(int n, int c, int h, int w);

Graph::NodeType MakeNewOperator(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops);

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor);

TensorParam Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                             const size_t iter_input_tensor, Graph::NodeType NewTensor);

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names);

void MakeEdge(const std::vector<std::vector<std::string>> &input_tensor_names, std::shared_ptr<Graph> graph);

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>> &input_tensor_names,
                                  const std::string &input_name);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
