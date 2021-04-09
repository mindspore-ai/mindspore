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

#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
const TensorParam MakeTensor(int64_t n, int64_t c, int64_t h, int64_t w) {
  TensorParam new_tensor;
  new_tensor.tensor_type = kFloat32;
  new_tensor.tensor_shape.shape_n = n;
  new_tensor.tensor_shape.shape_c = c;
  new_tensor.tensor_shape.shape_h = h;
  new_tensor.tensor_shape.shape_w = w;
  const TensorParam &tensor = new_tensor;
  return tensor;
}

Graph::NodeType MakeNewOperator(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops) {
  Graph::NodeType NewOp;
  NewOp.name = ops[iter_ops]->name();
  NewOp.info = InfoType::kApplication;

  auto op_type = ops[iter_ops]->type();
  auto idx = DictOpType.find(op_type);
  if (idx == DictOpType.end()) {
    NewOp.apply.op_type = OperatorType::kRecUnkownType;
    MS_LOG(INFO) << ops[iter_ops]->name() << ": Unknown operator type " << op_type;
  } else {
    NewOp.apply.op_type = DictOpType.at(op_type);
  }

  if (ops[iter_ops]->outputs_tensor_info().size() == 0) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " output tensor info is empty.";
  }

  if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 4) {
    NewOp.tensor_parm = MakeTensor(
      ops[iter_ops]->outputs_tensor_info()[0].shape()[0], ops[iter_ops]->outputs_tensor_info()[0].shape()[1],
      ops[iter_ops]->outputs_tensor_info()[0].shape()[2], ops[iter_ops]->outputs_tensor_info()[0].shape()[3]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 3) {
    NewOp.tensor_parm = MakeTensor(1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0],
                                   ops[iter_ops]->outputs_tensor_info()[0].shape()[1],
                                   ops[iter_ops]->outputs_tensor_info()[0].shape()[2]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 2) {
    NewOp.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0],
                                   ops[iter_ops]->outputs_tensor_info()[0].shape()[1]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 1) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0]);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 0) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    MS_LOG(ERROR) << ops[iter_ops]->name() << ": output tensor shape is unexpected.";
  }

  NewOp.apply = CompleteOperatorInputs(ops, iter_ops, NewOp);
  return NewOp;
}

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor) {
  size_t input_tensor_size = ops[iter_ops]->inputs_tensor_info().size();
  if (ops[iter_ops]->type() == STACK) {
    input_tensor_size = 1;
  }
  if (input_tensor_size > MAX_INPUT_NUM) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " input tensor num exceeds limit.";
  }

  for (size_t iter_input_tensors = 0; iter_input_tensors < input_tensor_size; iter_input_tensors++) {
    if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 4) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[2],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[3]);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 3) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[2]);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 2) {
      NewTensor.apply.arguments[iter_input_tensors] = Complete2DInputs(ops, iter_ops, iter_input_tensors, NewTensor);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 1) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 0) {
      NewTensor.apply.arguments[iter_input_tensors] = MakeTensor(1, 1, 1, 1);
    } else {
      MS_LOG(ERROR) << ops[iter_ops]->name() << ": input tensor shape is unexpected.";
    }
  }
  return NewTensor.apply;
}

TensorParam Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
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

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names) {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
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

void MakeEdge(const std::vector<std::vector<std::string>> &input_tensor_names, const std::shared_ptr<Graph> &graph) {
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

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>> &input_tensor_name,
                                  const std::string &input_name) {
  for (size_t index = 0; index < input_tensor_name.size(); index++) {
    if (input_tensor_name[index][0] == input_name) {
      return index;
    }
  }
  MS_LOG(INFO) << "Get index failed, using SIZE_MAX instead";
  return SIZE_MAX;
}

void Eliminate_Aux(const size_t node_index, const std::shared_ptr<Graph> &graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list) {
  std::vector<size_t> eli;
  eli.push_back(node_index);
  for (size_t i = 0; i < (size_t)graph->nodes[node_index].node_out.size(); i++) {
    eli.push_back(graph->nodes[node_index].node_out[i]);
  }
  eli_list->push_back(eli);

  for (size_t i = 0; i < graph->nodes[node_index].node_in.size(); i++) {
    auto *incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in[i]].node_out;
    auto it = find(incoming_outputs->begin(), incoming_outputs->end(), node_index);
    if (it != incoming_outputs->end()) {
      it = incoming_outputs->erase(it);
      incoming_outputs->insert(it, graph->nodes[node_index].node_out.begin(), graph->nodes[node_index].node_out.end());
    }
  }

  for (size_t i = 0; i < graph->nodes[node_index].node_in_aux.size(); i++) {
    auto *aux_incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in_aux[i]].node_out;
    auto it = find(aux_incoming_outputs->begin(), aux_incoming_outputs->end(), node_index);
    if (it != aux_incoming_outputs->end()) {
      it = aux_incoming_outputs->erase(it);
      aux_incoming_outputs->insert(it, graph->nodes[node_index].node_out.begin(),
                                   graph->nodes[node_index].node_out.end());
    }
  }

  for (size_t i = 0; i < graph->nodes[node_index].node_out.size(); i++) {
    auto *outgoing_inputs = &graph->nodes[graph->nodes[node_index].node_out[i]].node_in;
    auto it = find(outgoing_inputs->begin(), outgoing_inputs->end(), node_index);
    if (it != outgoing_inputs->end()) {
      if (graph->nodes[node_index].node_in.size() > 0) {
        outgoing_inputs->at(std::distance(outgoing_inputs->begin(), it)) = graph->nodes[node_index].node_in[0];
        for (size_t j = 1; j < graph->nodes[node_index].node_in.size(); j++) {
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.push_back(graph->nodes[node_index].node_in[j]);
        }
        for (size_t j = 1; j < graph->nodes[node_index].node_in_aux.size(); j++) {
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.push_back(
            graph->nodes[node_index].node_in_aux[j]);
        }
      } else {
        outgoing_inputs->erase(it);
      }
    }
  }
}

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> &graph,
                                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                      const std::shared_ptr<std::vector<size_t>> &index_list) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t node_index = 0; node_index < (size_t)graph->nodes.size(); node_index++) {
    auto type = graph->nodes[node_index].apply.op_type;
    if (ElementWiseOpType.find(type) != ElementWiseOpType.end()) {
      Eliminate_Aux(node_index, graph, eli_list);
    }
  }
  index_list->reserve(graph->nodes.size());
  for (size_t i = 0; i < (size_t)graph->nodes.size(); i++) {
    index_list->push_back(i);
  }
  for (size_t i = 0; i < (size_t)eli_list->size(); i++) {
    if (eli_list->at(i)[0] >= index_list->size()) {
      MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
    }
    index_list->at(eli_list->at(i)[0]) = SIZE_MAX;
    for (size_t j = eli_list->at(i)[0] + 1; j < (size_t)index_list->size(); j++) {
      index_list->at(j)--;
    }
  }
  std::shared_ptr<Graph> new_graph = std::make_shared<Graph>();
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    if (index_list->at(i) > SIZE_MAX / 2) {
      continue;
    }
    new_graph->nodes.push_back(graph->nodes[i]);
    auto *node_in = &new_graph->nodes[index_list->at(i)].node_in;
    for (size_t j = node_in->size(); j > 0; j--) {
      bool IsEliminated = (index_list->at(node_in->at(j - 1)) == SIZE_MAX);
      if (IsEliminated) {
        node_in->erase(node_in->begin() + j - 1);
      } else {
        node_in->at(j - 1) = index_list->at(node_in->at(j - 1));
      }
    }
    auto *node_out = &new_graph->nodes[index_list->at(i)].node_out;
    for (size_t j = node_out->size(); j > 0; j--) {
      bool IsEliminated = (index_list->at(node_out->at(j - 1)) == SIZE_MAX);
      if (IsEliminated) {
        node_out->erase(node_out->begin() + j - 1);
      } else {
        node_out->at(j - 1) = index_list->at(node_out->at(j - 1));
      }
    }
  }
  return new_graph;
}
}  // namespace parallel
}  // namespace mindspore
