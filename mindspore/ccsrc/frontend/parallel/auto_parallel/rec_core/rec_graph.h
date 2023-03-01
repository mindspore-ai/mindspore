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

#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "ir/anf.h"

namespace mindspore {
namespace parallel {
enum OperatorType {
  kRecUnknownType,
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
  kRecExpandDims,
  kRecStridedSlice,
  kRecArgWithValue,
  kRecUnsortedSegmentOp,
  kRecBatchMatMul
};

enum InfoType { kApplication, kConstant };

struct OperatorRec {
  OperatorType op_type;
  TensorParam arguments[MAX_INPUT_NUM];
  StrategyRec str;
  std::vector<StrategyRec> strs;
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
    // Nodes that point to this node via auxliary edges
    std::vector<size_t> node_in_aux;

    // Node Type Info: Application or Constant. Defined in enum <InfoType> .
    InfoType info;
    // Operator info. Defined in struct <OperatorRec> .
    OperatorRec apply;
    // Tensor info. Defined in tensor.h struct <TensorParam> .
    TensorParam tensor_parm;
  };

  int64_t batch_size;

  std::vector<Graph::NodeType> nodes;  // Nodes of the graph. Public.
};                                     // Define simplified dataflow Graph for partitioning
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GRAPH_H_
