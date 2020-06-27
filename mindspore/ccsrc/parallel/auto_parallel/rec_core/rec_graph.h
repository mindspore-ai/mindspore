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

#ifndef PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_
#define PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_

#include <iostream>
#include <string>
#include <vector>

#include "parallel/auto_parallel/rec_core/rec_strategy.h"
#include "parallel/auto_parallel/rec_core/rec_tensor.h"

namespace mindspore {
namespace parallel {
enum OperatorType {
  kRecUnkownType,
  kRecMatMul,
  kRecConvolution,
  kRecPooling,
  kRecElmWiseOp,
  kRecReLU,
  kRecBatchNorm,
  kRecReshape,
  kRecBiasAdd,
  kRecSoftmax,
  kRecSparseSoftmaxCrossEntropyWithLogits,
  kRecSoftmaxCrossEntropyWithLogits,
  kRecOneHot,
  kRecLog,
  kRecExp,
  kRecAdd,
  kRecSub,
  kRecMul,
  kRecDiv,
  kRecSqueeze,
  kRecCast,
  kRecReduce,
  kRecPReLU,
  kRecGatherV2,
  kRecArgWithValue
};

enum InfoType { kApplication, kConstant };

struct OperatorRec {
  OperatorType op_type;
  TensorParam arguments[MAX_INPUT_NUM];
  StrategyRec str;
};

// Define simplified dataflow Graph for partitioning
class Graph {
 public:
  struct NodeType {
    std::string name;
    // Nodes that point to this node
    std::vector<size_t> node_in;
    // Nodes that point from this node
    std::vector<size_t> node_out;
    std::vector<size_t> node_in_aux;
    // Node Type Info: Application or Constant. Defined in enum <InfoType> .
    InfoType info;
    // Operator info. Defined in struct <OperatorRec> .
    OperatorRec apply;
    // Tensor info. Defined in tensor.h struct <TensorParam> .
    TensorParam tensor_parm;
  };

  std::vector<Graph::NodeType> nodes;  // Nodes of the graph. Pubic.
};                                     // Define simplified dataflow Graph for partitioning
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_
