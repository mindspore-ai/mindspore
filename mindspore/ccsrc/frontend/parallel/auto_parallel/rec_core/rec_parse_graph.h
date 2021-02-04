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
  OperatorType::kRecReLU,      OperatorType::kRecLog,      OperatorType::kRecExp,         OperatorType::kRecAdd,
  OperatorType::kRecElmWiseOp, OperatorType::kRecBiasAdd,  OperatorType::kRecSub,         OperatorType::kRecMul,
  OperatorType::kRecDiv,       OperatorType::kRecSqueeze,  OperatorType::kRecReduce,      OperatorType::kRecCast,
  OperatorType::kRecReshape,   OperatorType::kRecGatherV2, OperatorType::kRecArgWithValue};

const std::map<std::string, OperatorType> DictOpType{
  {MATMUL, OperatorType::kRecMatMul},
  {CONV2D, OperatorType::kRecConvolution},
  {MAXPOOL, OperatorType::kRecPooling},
  {MAXPOOLV2, OperatorType::kRecPooling},
  {POOLING, OperatorType::kRecPooling},
  {MAX_POOL_WITH_ARGMAX, OperatorType::kRecPooling},
  {SIMPLE_MEAN, OperatorType::kRecPooling},
  {RESHAPE, OperatorType::kRecReshape},
  {BIAS_ADD, OperatorType::kRecBiasAdd},
  {BATCH_NORM, OperatorType::kRecBatchNorm},
  {LAYER_NORM, OperatorType::kRecBatchNorm},
  {SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits},
  {ONEHOT, OperatorType::kRecOneHot},
  {SQUEEZE, OperatorType::kRecSqueeze},
  {CAST, OperatorType::kRecCast},
  {REDUCE_SUM, OperatorType::kRecReduce},
  {REDUCE_MAX, OperatorType::kRecReduce},
  {REDUCE_MIN, OperatorType::kRecReduce},
  {REDUCE_MEAN, OperatorType::kRecReduce},
  {GATHERV2, OperatorType::kRecGatherV2},
  {ARGMAXWITHVALUE, OperatorType::kRecArgWithValue},
  {ARGMINWITHVALUE, OperatorType::kRecArgWithValue},
  {UNSORTED_SEGMENT_SUM, OperatorType::kRecUnsortedSegmentOp},
  {UNSORTED_SEGMENT_MAX, OperatorType::kRecUnsortedSegmentOp},
  {UNSORTED_SEGMENT_MIN, OperatorType::kRecUnsortedSegmentOp},
  // Activation OP
  {ACTIVATION, OperatorType::kRecReLU},
  {RELU, OperatorType::kRecReLU},
  {"ReLU6", OperatorType::kRecReLU},
  {"ReLUV2", OperatorType::kRecReLU},
  {SIGMOID, OperatorType::kRecReLU},
  {SIGMOID_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecReLU},
  {"HSigmoid", OperatorType::kRecReLU},
  {GELU, OperatorType::kRecReLU},
  {TANH, OperatorType::kRecReLU},
  {SOFTPLUS, OperatorType::kRecReLU},
  {SOFTSIGN, OperatorType::kRecReLU},
  {PRELU, OperatorType::kRecPReLU},
  // Elm-wise OP
  {TRANSPOSE, OperatorType::kRecElmWiseOp},
  {L2_NORMALIZE, OperatorType::kRecElmWiseOp},
  {ADD, OperatorType::kRecElmWiseOp},
  {TENSOR_DOT, OperatorType::kRecElmWiseOp},
  {SUB, OperatorType::kRecElmWiseOp},
  {MUL, OperatorType::kRecElmWiseOp},
  {DIV, OperatorType::kRecElmWiseOp},
  {REAL_DIV, OperatorType::kRecElmWiseOp},
  {SOFTMAX, OperatorType::kRecSoftmax},
  {LOG_SOFTMAX, OperatorType::kRecSoftmax},
  {SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, OperatorType::kRecSoftmaxCrossEntropyWithLogits},
  {SQRT, OperatorType::kRecElmWiseOp},
  {NEG, OperatorType::kRecElmWiseOp},
  {POW, OperatorType::kRecElmWiseOp},
  {EXP, OperatorType::kRecElmWiseOp},
  {LOG, OperatorType::kRecElmWiseOp},
  {COS, OperatorType::kRecElmWiseOp},
  {ACOS, OperatorType::kRecElmWiseOp},
  {ASIN, OperatorType::kRecElmWiseOp},
  {ASINH, OperatorType::kRecElmWiseOp},
  {ATAN, OperatorType::kRecElmWiseOp},
  {ATANH, OperatorType::kRecElmWiseOp},
  {EXPM1, OperatorType::kRecElmWiseOp},
  {LOG1P, OperatorType::kRecElmWiseOp},
  {LOGICALNOT, OperatorType::kRecElmWiseOp},
  {"LogicalAnd", OperatorType::kRecElmWiseOp},
  {"LogicalOr", OperatorType::kRecElmWiseOp},
  {SQUARE, OperatorType::kRecElmWiseOp},
  {"Abs", OperatorType::kRecElmWiseOp},
  {"Acosh", OperatorType::kRecElmWiseOp},
  {"AddN", OperatorType::kRecElmWiseOp},
  {"AccumulateNV2", OperatorType::kRecElmWiseOp},
  {"Atan2", OperatorType::kRecElmWiseOp},
  {ELU, OperatorType::kRecElmWiseOp},
  {ERF, OperatorType::kRecElmWiseOp},
  {ERFC, OperatorType::kRecElmWiseOp},
  {MOD, OperatorType::kRecElmWiseOp},
  {FLOOR, OperatorType::kRecElmWiseOp},
  {CEIL, OperatorType::kRecElmWiseOp},
  {FLOORDIV, OperatorType::kRecElmWiseOp},
  {"FloorMod", OperatorType::kRecElmWiseOp},
  {GREATER, OperatorType::kRecElmWiseOp},
  {"GreaterEqual", OperatorType::kRecElmWiseOp},
  {"HSwish", OperatorType::kRecElmWiseOp},
  {"Less", OperatorType::kRecElmWiseOp},
  {"LessEqual", OperatorType::kRecElmWiseOp},
  {MAXIMUM, OperatorType::kRecElmWiseOp},
  {MINIMUM, OperatorType::kRecElmWiseOp},
  {EQUAL, OperatorType::kRecElmWiseOp},
  {NOT_EQUAL, OperatorType::kRecElmWiseOp},
  {APPROXIMATEEQUAL, OperatorType::kRecElmWiseOp},
  {INV, OperatorType::kRecElmWiseOp},
  {BESSELI0E, OperatorType::kRecElmWiseOp},
  {BESSELI1E, OperatorType::kRecElmWiseOp},
  {ZEROSLIKE, OperatorType::kRecElmWiseOp},
  {ONESLIKE, OperatorType::kRecElmWiseOp},
  {DIVNONAN, OperatorType::kRecElmWiseOp},
  {"Reciprocal", OperatorType::kRecElmWiseOp},
  {"Round", OperatorType::kRecElmWiseOp},
  {"Rsqrt", OperatorType::kRecElmWiseOp},
  {"Sign", OperatorType::kRecElmWiseOp},
  {SIN, OperatorType::kRecElmWiseOp},
  {SINH, OperatorType::kRecElmWiseOp},
  {TAN, OperatorType::kRecElmWiseOp},
  {ASSIGN, OperatorType::kRecElmWiseOp},
  {ASSIGN_ADD, OperatorType::kRecElmWiseOp},
  {ASSIGN_SUB, OperatorType::kRecElmWiseOp},
  {"AssignAdd", OperatorType::kRecElmWiseOp},
  {DROPOUT_DO_MASK, OperatorType::kRecElmWiseOp},
  {STACK, OperatorType::kRecElmWiseOp}};

const TensorParam MakeTensor(int64_t n, int64_t c, int64_t h, int64_t w);

Graph::NodeType MakeNewOperator(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops);

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor);

TensorParam Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                             const size_t iter_input_tensor, Graph::NodeType NewTensor);

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names);

void MakeEdge(const std::vector<std::vector<std::string>> &input_tensor_names, const std::shared_ptr<Graph> &graph);

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>> &input_tensor_names,
                                  const std::string &input_name);

void Eliminate_Aux(const size_t node_index, const std::shared_ptr<Graph> &graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list);

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> &graph,
                                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                      const std::shared_ptr<std::vector<size_t>> &index_list);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_PARSE_GRAPH_H_
