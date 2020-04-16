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

#include "parallel/auto_parallel/rec_core/rec_parse_graph.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ir/value.h"
#include "parallel/auto_parallel/rec_core/rec_graph.h"
#include "parallel/auto_parallel/rec_core/rec_tensor.h"
#include "parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
const TensorParam MakeTensor(int n, int c, int h, int w) {
  TensorParam new_tensor;
  new_tensor.tensor_type = kFloat32;
  new_tensor.tensor_shape.shape_n = n;
  new_tensor.tensor_shape.shape_c = c;
  new_tensor.tensor_shape.shape_h = h;
  new_tensor.tensor_shape.shape_w = w;
  const TensorParam& tensor = new_tensor;
  return tensor;
}

Graph::NodeType MakeNewOperator(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops) {
  Graph::NodeType NewOp;
  NewOp.name = ops[iter_ops]->name();
  NewOp.info = InfoType::kApplication;

  auto op_type = ops[iter_ops]->type();
  auto idx = DictOpType.find(op_type);
  if (idx == DictOpType.end()) {
    NewOp.apply.op_type = OperatorType::kRecUnkownType;
    MS_LOG(INFO) << "Unknown operator type.";
  } else {
    NewOp.apply.op_type = DictOpType.at(op_type);
  }

  if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 4) {
    NewOp.tensor_parm = MakeTensor(
      ops[iter_ops]->outputs_tensor_info()[0].shape()[0], ops[iter_ops]->outputs_tensor_info()[0].shape()[1],
      ops[iter_ops]->outputs_tensor_info()[0].shape()[2], ops[iter_ops]->outputs_tensor_info()[0].shape()[3]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 2) {
    NewOp.tensor_parm = Fill2DTensor(ops, iter_ops, NewOp);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 1) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 0) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    MS_LOG(ERROR) << "Tensor's shape is unknown.";
  }

  NewOp.apply = CompleteOperatorInputs(ops, iter_ops, NewOp);
  return NewOp;
}

TensorParam Fill2DTensor(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                         Graph::NodeType NewTensor) {
  if (NewTensor.apply.op_type == OperatorType::kRecMatMul) {
    auto attrs = ops[iter_ops]->attrs();
    bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
    bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
    if (transpose_a) {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[0].shape()[1],
                                         ops[iter_ops]->inputs_tensor_info()[0].shape()[0]);
    } else if (transpose_b) {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[0].shape()[1],
                                         ops[iter_ops]->inputs_tensor_info()[0].shape()[0]);
    } else {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[0].shape()[0],
                                         ops[iter_ops]->inputs_tensor_info()[0].shape()[1]);
    }
  } else {
    NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[0].shape()[0],
                                       ops[iter_ops]->inputs_tensor_info()[0].shape()[1]);
  }
  return NewTensor.tensor_parm;
}

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor) {
  for (size_t iter_input_tensors = 0; iter_input_tensors < ops[iter_ops]->inputs_tensor_info().size();
       iter_input_tensors++) {
    if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 4) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[2],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[3]);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 2) {
      NewTensor.apply.arguments[iter_input_tensors] = Complete2DInputs(ops, iter_ops, iter_input_tensors, NewTensor);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 1) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 0) {
      NewTensor.apply.arguments[iter_input_tensors] = MakeTensor(1, 1, 1, 1);
    } else {
      MS_LOG(ERROR) << "Tensor's shape is unknown.";
    }
  }
  return NewTensor.apply;
}

TensorParam Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                             const size_t iter_input_tensors, Graph::NodeType NewTensor) {
  if (NewTensor.apply.op_type == OperatorType::kRecMatMul) {
    auto attrs = ops[iter_ops]->attrs();
    bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
    bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_input_tensors == 0)) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
    }
  } else {
    NewTensor.apply.arguments[iter_input_tensors] =
      MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                 ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
  }
  return NewTensor.apply.arguments[iter_input_tensors];
}

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>>& ops,
                                  const std::vector<std::vector<std::string>>& input_tensor_names) {
  std::shared_ptr<Graph> graph(new Graph);
  if (ops.size() > SIZE_MAX / 2) {
    MS_LOG(EXCEPTION) << "Total number of operators is bigger than " << SIZE_MAX / 2;
  }

  for (size_t iter_ops = 0; iter_ops < ops.size(); iter_ops++) {
    Graph::NodeType NewOp = MakeNewOperator(ops, iter_ops);
    graph->nodes.push_back(NewOp);
  }
  MakeEdge(input_tensor_names, graph);

  return graph;
}

void MakeEdge(const std::vector<std::vector<std::string>>& input_tensor_names, std::shared_ptr<Graph> graph) {
  for (size_t iter_i = 0; iter_i < input_tensor_names.size(); iter_i++) {
    for (size_t iter_j = 1; iter_j < input_tensor_names[iter_i].size(); iter_j++) {
      size_t head_node_index = GetIndexInInputTensorNames(input_tensor_names, input_tensor_names[iter_i][iter_j]);
      if (head_node_index < SIZE_MAX / 2 && head_node_index != iter_i) {
        graph->nodes[iter_i].node_in.push_back(head_node_index);
        graph->nodes[head_node_index].node_out.push_back(iter_i);
      }
    }
  }
}

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>>& input_tensor_name,
                                  const std::string& input_name) {
  for (size_t index = 0; index < input_tensor_name.size(); index++) {
    if (input_tensor_name[index][0] == input_name) {
      return index;
    }
  }
  MS_LOG(INFO) << "Get index failed, using SIZE_MAX insted";
  return SIZE_MAX;
}
}  // namespace parallel
}  // namespace mindspore
