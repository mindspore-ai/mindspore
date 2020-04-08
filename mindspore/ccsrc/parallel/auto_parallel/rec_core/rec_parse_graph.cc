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

bool IsInList(const std::string& name, const std::vector<std::string>& list) {
  return std::find(list.begin(), list.end(), name) != list.end();
}

Graph::NodeType MakeNewOperator(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops) {
  Graph::NodeType NewOp;
  NewOp.name = ops[iter_ops]->cnode_name();
  NewOp.info = InfoType::kApplication;

  auto op_type = ops[iter_ops]->type();
  auto idx = DictOpType.find(op_type);
  if (idx == DictOpType.end()) {
    NewOp.apply.op_type = OperatorType::kRecUnkownType;
    MS_LOG(INFO) << "Unknown type in rec_parse_graph::MakeNewOperator";
  } else {
    NewOp.apply.op_type = DictOpType.at(op_type);
  }

  if ((NewOp.apply.op_type == OperatorType::kRecMatMul) || (NewOp.apply.op_type == OperatorType::kRecBiasAdd) ||
      (NewOp.apply.op_type == OperatorType::kRecReshape)) {
    NewOp.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0],
                                   ops[iter_ops]->outputs_tensor_info()[0].shape()[1]);
  } else if ((NewOp.apply.op_type == OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits) ||
             (NewOp.apply.op_type == OperatorType::kRecUnkownType)) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    NewOp.tensor_parm = MakeTensor(
      ops[iter_ops]->outputs_tensor_info()[0].shape()[0], ops[iter_ops]->outputs_tensor_info()[0].shape()[1],
      ops[iter_ops]->outputs_tensor_info()[0].shape()[2], ops[iter_ops]->outputs_tensor_info()[0].shape()[3]);
  }

  return NewOp;
}

Graph::NodeType MakeNewTensor(std::vector<std::shared_ptr<OperatorInfo>> ops, const size_t iter_ops,
                              const std::string& input, const size_t iter_input_tensors, std::shared_ptr<Graph> graph,
                              size_t current_op_index) {
  Graph::NodeType NewTensor;
  NewTensor.name = input;
  NewTensor.info = InfoType::kConstant;

  if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 4) {
    NewTensor.tensor_parm = MakeTensor(ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                                       ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                                       ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[2],
                                       ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[3]);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 2) {
    Fill2DTensor(ops, iter_ops, graph, iter_input_tensors, current_op_index, NewTensor);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 1) {
    NewTensor.tensor_parm = MakeTensor(1, 1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 0) {
    NewTensor.tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    MS_LOG(ERROR) << "Tensor's shape unknown in rec_parse_graph::MakeNewTensor";
  }
  return NewTensor;
}

void Fill2DTensor(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                  const std::shared_ptr<Graph> graph, const size_t iter_input_tensors, const size_t current_op_index,
                  Graph::NodeType NewTensor) {
  if (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecMatMul) {
    auto attrs = ops[iter_ops]->attrs();
    bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
    bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_input_tensors == 0)) {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                                         ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                                         ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else {
      NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                                         ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
    }
  } else {
    NewTensor.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                                       ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
  }
}

void CompleteOperatorInputs(std::vector<std::shared_ptr<OperatorInfo>> ops, size_t iter_ops, size_t iter_input_tensors,
                            size_t current_op_index, std::shared_ptr<Graph> graph) {
  if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 4) {
    graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
      MakeTensor(ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                 ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                 ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[2],
                 ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[3]);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 2) {
    Complete2DInputs(ops, iter_ops, graph, iter_input_tensors, current_op_index);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 1) {
    graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
      MakeTensor(1, 1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
  } else if (ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape().size() == 0) {
    graph->nodes[current_op_index].apply.arguments[iter_input_tensors] = MakeTensor(1, 1, 1, 1);
  } else {
    MS_LOG(ERROR) << "Tensor's shape unknown in rec_parse_graph::MakeNewTensor";
  }
}

void Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>>& ops, const size_t iter_ops,
                      const std::shared_ptr<Graph> graph, const size_t iter_input_tensors,
                      const size_t current_op_index) {
  if (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecMatMul) {
    auto attrs = ops[iter_ops]->attrs();
    bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
    bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_input_tensors == 0)) {
      graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0]);
    } else {
      graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                   ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
    }
  } else {
    graph->nodes[current_op_index].apply.arguments[iter_input_tensors] =
      MakeTensor(1, 1, ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[0],
                 ops[iter_ops]->inputs_tensor_info()[iter_input_tensors].shape()[1]);
  }
}

void MakeEdge(std::shared_ptr<Graph> graph, const size_t input_index, const size_t current_op_index) {
  graph->nodes[input_index].node_out.push_back(current_op_index);
  graph->nodes[current_op_index].node_in.push_back(input_index);
}

void ModifyTensorToOperator(std::shared_ptr<Graph> graph, const size_t current_op_index, const size_t iter_ops,
                            std::vector<std::shared_ptr<OperatorInfo>> ops) {
  graph->nodes[current_op_index].info = InfoType::kApplication;
  std::string op_type = ops[iter_ops]->type();
  auto idx = DictOpType.find(op_type);
  if (idx == DictOpType.end()) {
    graph->nodes[current_op_index].apply.op_type = OperatorType::kRecUnkownType;
    MS_LOG(INFO) << "Unknown type in rec_parse_graph::ModifyTensorToOperator";
  } else {
    graph->nodes[current_op_index].apply.op_type = DictOpType.at(op_type);
  }

  if ((graph->nodes[current_op_index].apply.op_type == OperatorType::kRecMatMul) ||
      (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecBiasAdd) ||
      (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecReshape)) {
    graph->nodes[current_op_index].tensor_parm = MakeTensor(1, 1, ops[iter_ops]->outputs_tensor_info()[0].shape()[0],
                                                            ops[iter_ops]->outputs_tensor_info()[0].shape()[1]);
  } else if ((graph->nodes[current_op_index].apply.op_type == OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits) ||
             (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecUnkownType)) {
    graph->nodes[current_op_index].tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    graph->nodes[current_op_index].tensor_parm = MakeTensor(
      ops[iter_ops]->outputs_tensor_info()[0].shape()[0], ops[iter_ops]->outputs_tensor_info()[0].shape()[1],
      ops[iter_ops]->outputs_tensor_info()[0].shape()[2], ops[iter_ops]->outputs_tensor_info()[0].shape()[3]);
  }
}

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>>& ops,
                                  const std::vector<std::vector<std::string>>& input_tensor_names,
                                  const std::shared_ptr<std::vector<size_t>>& ops_nodes_list) {
  std::vector<std::string> current_graph;
  std::shared_ptr<Graph> graph(new Graph);
  if (ops.size() > SIZE_MAX / 2) {
    MS_LOG(EXCEPTION) << "Total number of operators is bigger than " << SIZE_MAX / 2;
  }

  for (size_t iter_ops = ops.size(); iter_ops > 0; iter_ops--) {
    if (IsInList(ops[iter_ops - 1]->cnode_name(), current_graph)) {
      size_t current_op_index = static_cast<size_t>(std::distance(
        current_graph.begin(), std::find(current_graph.begin(), current_graph.end(), ops[iter_ops]->cnode_name())));
      std::vector<size_t>::iterator itr = ops_nodes_list->insert(ops_nodes_list->begin(), current_op_index);
      if (itr != ops_nodes_list->begin()) {
        MS_LOG(EXCEPTION) << "Iterator error.";
      }
      ModifyTensorToOperator(graph, current_op_index, iter_ops - 1, ops);
      LinkOps(graph, ops, input_tensor_names, current_graph, iter_ops - 1, current_op_index);
    } else {
      Graph::NodeType NewOp = MakeNewOperator(ops, iter_ops - 1);
      current_graph.push_back(NewOp.name);
      graph->nodes.push_back(NewOp);
      size_t current_op_index = graph->nodes.size() - 1;
      std::vector<size_t>::iterator itr = ops_nodes_list->insert(ops_nodes_list->begin(), current_op_index);
      if (itr != ops_nodes_list->begin()) {
        MS_LOG(EXCEPTION) << "Iterator error.";
      }
      LinkOps(graph, ops, input_tensor_names, current_graph, iter_ops - 1, current_op_index);
    }
  }
  return graph;
}

void LinkOps(std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<OperatorInfo>> ops,
             const std::vector<std::vector<std::string>>& input_tensor_names, std::vector<std::string> current_graph,
             const size_t iter_ops, const size_t current_op_index) {
  for (size_t iter_input_tensors = 0;
       iter_input_tensors < std::min(input_tensor_names[iter_ops].size(), ops[iter_ops]->inputs_tensor_info().size());
       iter_input_tensors++) {
    std::string input = input_tensor_names[iter_ops][iter_input_tensors];
    if (IsInList(input, current_graph)) {
      size_t input_index = static_cast<size_t>(
        std::distance(current_graph.begin(), std::find(current_graph.begin(), current_graph.end(), input)));
      MakeEdge(graph, input_index, current_op_index);
      CompleteOperatorInputs(ops, iter_ops, iter_input_tensors, current_op_index, graph);
    } else {
      Graph::NodeType NewTensor = MakeNewTensor(ops, iter_ops, input, iter_input_tensors, graph, current_op_index);
      current_graph.push_back(NewTensor.name);
      graph->nodes.push_back(NewTensor);
      size_t input_index = graph->nodes.size() - 1;
      CompleteOperatorInputs(ops, iter_ops, iter_input_tensors, current_op_index, graph);
      MakeEdge(graph, input_index, current_op_index);
    }

    if (graph->nodes[current_op_index].apply.op_type == OperatorType::kRecBatchNorm) {
      break;
    }
  }
}

void Eliminate_Aux(const size_t node_index, std::shared_ptr<Graph> graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list) {
  if ((graph->nodes[node_index].apply.op_type == OperatorType::kRecUnkownType) ||
      (graph->nodes[node_index].apply.op_type == OperatorType::kRecReLU)) {
    size_t input_index = (graph->nodes[node_index].node_in)[0];
    std::vector<size_t> outputs = graph->nodes[node_index].node_out;

    std::vector<size_t> eli;
    eli.push_back(node_index);
    eli.push_back(input_index);
    for (size_t i = 0; i < outputs.size(); i++) {
      eli.push_back(i);
    }
    eli_list->push_back(eli);

    for (size_t i = 1; i < (size_t)graph->nodes[node_index].node_in.size(); i++) {
      std::vector<size_t> tmp;
      tmp.push_back(node_index);
      tmp.push_back((graph->nodes[node_index].node_in)[i]);
      eli_list->push_back(tmp);
    }

    auto it = find(graph->nodes[input_index].node_out.begin(), graph->nodes[input_index].node_out.end(), node_index);
    std::vector<size_t>::iterator itr = graph->nodes[input_index].node_out.erase(it);
    if (itr != it) {
      MS_LOG(EXCEPTION) << "Iterator error.";
    }
    for (auto output : outputs) {
      graph->nodes[input_index].node_out.push_back(output);
    }
    for (auto& output_index : outputs) {
      auto itt = find(graph->nodes[output_index].node_in.begin(), graph->nodes[output_index].node_in.end(), node_index);
      graph->nodes[output_index]
        .node_in[static_cast<size_t>(std::distance(graph->nodes[output_index].node_in.begin(), itt))] = input_index;
    }
  }
}

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> graph,
                                      std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                                      std::shared_ptr<std::vector<size_t>> index_list) {
  for (size_t node_index = 0; node_index < (size_t)graph->nodes.size(); node_index++) {
    if (graph->nodes[node_index].info == InfoType::kApplication) {
      Eliminate_Aux(node_index, graph, eli_list);
    }
  }

  index_list->reserve(graph->nodes.size());
  for (size_t i = 0; i < (size_t)graph->nodes.size(); i++) {
    index_list->push_back(i);
  }

  for (size_t i = 0; i < (size_t)eli_list->size(); i++) {
    index_list->at((eli_list->at(i)[0])) = SIZE_MAX;
    for (size_t j = eli_list->at(i)[0] + 1; j < (size_t)index_list->size(); j++) {
      index_list->at(j)--;
    }
  }

  std::shared_ptr<Graph> new_graph(new Graph);
  for (size_t i = 0; i < (size_t)(graph->nodes.size() - eli_list->size()); i++) {
    Graph::NodeType NewOp;
    new_graph->nodes.push_back(NewOp);
  }

  for (size_t i = 0; i < (size_t)graph->nodes.size(); i++) {
    if (index_list->at(i) > SIZE_MAX / 2) continue;
    new_graph->nodes[index_list->at(i)] = graph->nodes[i];
    for (size_t j = 0; j < (size_t)new_graph->nodes[index_list->at(i)].node_in.size(); j++) {
      new_graph->nodes[index_list->at(i)].node_in[j] = index_list->at(new_graph->nodes[index_list->at(i)].node_in[j]);
    }
    for (size_t j = 0; j < (size_t)new_graph->nodes[index_list->at(i)].node_out.size(); j++) {
      new_graph->nodes[index_list->at(i)].node_out[j] = index_list->at(new_graph->nodes[index_list->at(i)].node_out[j]);
    }
  }

  return new_graph;
}
}  // namespace parallel
}  // namespace mindspore
