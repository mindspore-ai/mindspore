/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
    NewOp.apply.op_type = OperatorType::kRecUnknownType;
    MS_LOG(INFO) << ops[iter_ops]->name() << ": Unknown operator type " << op_type;
  } else {
    NewOp.apply.op_type = DictOpType.at(op_type);
  }

  if (ops[iter_ops]->outputs_shape().size() == SIZE_ZERO) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " outputs shape is empty.";
  }

  if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_FOUR) {
    NewOp.tensor_parm = MakeTensor(ops[iter_ops]->outputs_shape()[0][0], ops[iter_ops]->outputs_shape()[0][1],
                                   ops[iter_ops]->outputs_shape()[INDEX_ZERO][INDEX_TWO],
                                   ops[iter_ops]->outputs_shape()[INDEX_ZERO][INDEX_THREE]);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_THREE) {
    NewOp.tensor_parm = MakeTensor(1, ops[iter_ops]->outputs_shape()[0][0], ops[iter_ops]->outputs_shape()[0][1],
                                   ops[iter_ops]->outputs_shape()[INDEX_ZERO][INDEX_TWO]);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_TWO) {
    NewOp.tensor_parm = MakeTensor(1, 1, ops[iter_ops]->outputs_shape()[0][0], ops[iter_ops]->outputs_shape()[0][1]);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_ONE) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, ops[iter_ops]->outputs_shape()[0][0]);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_ZERO) {
    NewOp.tensor_parm = MakeTensor(1, 1, 1, 1);
  } else {
    MS_LOG(ERROR) << ops[iter_ops]->name() << ": output tensor shape is unexpected.";
  }

  NewOp.apply = CompleteOperatorInputs(ops, iter_ops, NewOp);
  return NewOp;
}

OperatorRec CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                   Graph::NodeType NewTensor) {
  size_t input_tensor_size = ops[iter_ops]->inputs_shape().size();
  if (ops[iter_ops]->type() == STACK) {
    input_tensor_size = 1;
  }
  if (input_tensor_size > MAX_INPUT_NUM) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " input tensor num exceeds limit.";
  }

  for (size_t iter_input_tensors = 0; iter_input_tensors < input_tensor_size; iter_input_tensors++) {
    if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_FOUR) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_THREE]);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_THREE) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO]);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_TWO) {
      NewTensor.apply.arguments[iter_input_tensors] = Complete2DInputs(ops, iter_ops, iter_input_tensors, NewTensor);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_ONE) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO]);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == 0) {
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
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][1],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][0]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][1],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][0]);
    } else {
      NewTensor.apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][0],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][1]);
    }
  } else {
    NewTensor.apply.arguments[iter_input_tensors] = MakeTensor(
      1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][0], ops[iter_ops]->inputs_shape()[iter_input_tensors][1]);
  }
  return NewTensor.apply.arguments[iter_input_tensors];
}

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names) {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  constexpr size_t MAX_OP_NUM = SIZE_MAX / 2;
  if (ops.size() > MAX_OP_NUM) {
    MS_LOG(EXCEPTION) << "Total number of operators is bigger than " << MAX_OP_NUM;
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

void Eliminate_Aux(size_t node_index, const std::shared_ptr<Graph> &graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list) {
  std::vector<size_t> eli;
  eli.push_back(node_index);
  for (size_t i = 0; i < graph->nodes[node_index].node_out.size(); i++) {
    eli.push_back(graph->nodes[node_index].node_out[i]);
  }
  eli_list->push_back(eli);

  for (size_t i = 0; i < graph->nodes[node_index].node_in.size(); i++) {
    auto *incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in[i]].node_out;
    auto it = find(incoming_outputs->begin(), incoming_outputs->end(), node_index);
    if (it != incoming_outputs->end()) {
      it = incoming_outputs->erase(it);
      for (auto outgoing_index : graph->nodes[node_index].node_out) {
        it = find(incoming_outputs->begin(), incoming_outputs->end(), outgoing_index);
        if (it == incoming_outputs->end()) {
          incoming_outputs->push_back(outgoing_index);
        }
      }
    }
  }

  for (size_t i = 0; i < graph->nodes[node_index].node_in_aux.size(); i++) {
    auto *aux_incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in_aux[i]].node_out;
    auto it = find(aux_incoming_outputs->begin(), aux_incoming_outputs->end(), node_index);
    if (it != aux_incoming_outputs->end()) {
      it = aux_incoming_outputs->erase(it);
      for (auto outgoing_index : graph->nodes[node_index].node_out) {
        it = find(aux_incoming_outputs->begin(), aux_incoming_outputs->end(), outgoing_index);
        if (it == aux_incoming_outputs->end()) {
          aux_incoming_outputs->push_back(outgoing_index);
        }
      }
    }
  }
  Eliminate_Aux_Outgoing(node_index, graph);
}

void EliminateAuxOutgoingInput(size_t node_index, const std::shared_ptr<Graph> &graph, size_t i) {
  auto *outgoing_inputs = &graph->nodes[graph->nodes[node_index].node_out[i]].node_in;
  auto it = find(outgoing_inputs->begin(), outgoing_inputs->end(), node_index);
  if (it != outgoing_inputs->end()) {
    if (graph->nodes[node_index].node_in.size() > 0) {
      auto exist_in_outgoing_auxinputs =
        find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.begin(),
             graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end(), graph->nodes[node_index].node_in[0]);
      if (exist_in_outgoing_auxinputs != graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end()) {
        size_t index_remove_node = LongToSize(std::distance(
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.begin(), exist_in_outgoing_auxinputs));
        if (graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx.size() > index_remove_node) {
          (void)graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx.erase(
            graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx.begin() + index_remove_node);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove_node << ", out of range!";
        }
        if (graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.size() > index_remove_node) {
          (void)graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.erase(exist_in_outgoing_auxinputs);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove_node
                        << ", which is out of range!";
        }
      }
      size_t idx = LongToSize(std::distance(outgoing_inputs->begin(), it));
      if (outgoing_inputs->size() > idx) {
        outgoing_inputs->at(idx) = graph->nodes[node_index].node_in[0];
      } else {
        MS_LOG(DEBUG) << "Trying to index vector element at index " << idx << ", out of range!";
      }

      for (size_t j = 1; j < graph->nodes[node_index].node_in.size(); j++) {
        exist_in_outgoing_auxinputs = find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.begin(),
                                           graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end(),
                                           graph->nodes[node_index].node_in[j]);
        if (exist_in_outgoing_auxinputs == graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end()) {
          size_t index_aux = LongToSize(std::distance(outgoing_inputs->begin(), it));
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx.push_back(index_aux);
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.push_back(graph->nodes[node_index].node_in[j]);
        }
      }
      for (size_t j = 0; j < graph->nodes[node_index].node_in_aux.size(); j++) {
        exist_in_outgoing_auxinputs = find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.begin(),
                                           graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end(),
                                           graph->nodes[node_index].node_in_aux[j]);
        if (exist_in_outgoing_auxinputs == graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.end()) {
          size_t index_aux = LongToSize(std::distance(outgoing_inputs->begin(), it));
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx.push_back(index_aux);
          graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux.push_back(
            graph->nodes[node_index].node_in_aux[j]);
        }
      }
    } else {
      auto idx = LongToSize(std::distance(outgoing_inputs->begin(), it));
      if (outgoing_inputs->size() > idx) {
        (void)outgoing_inputs->erase(it);
      } else {
        MS_LOG(DEBUG) << "Trying to erase vector element at index " << idx << ", out of range!";
      }
    }
  }
}

void EliminateAuxOutgoingAuxInput(size_t node_index, const std::shared_ptr<Graph> &graph, size_t i) {
  auto *outgoing_auxinputs = &graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux;
  auto *outgoing_auxinputs_index = &graph->nodes[graph->nodes[node_index].node_out[i]].node_in_aux_idx;
  auto it = find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), node_index);
  size_t index_entree = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
  if (it != outgoing_auxinputs->end()) {
    if (graph->nodes[node_index].node_in.size() > 0) {
      auto exist_in_outgoing_inputs =
        find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in.begin(),
             graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end(), graph->nodes[node_index].node_in[0]);
      if (exist_in_outgoing_inputs != graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end()) {
        index_entree = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
        if (outgoing_auxinputs_index->size() > index_entree) {
          (void)outgoing_auxinputs_index->erase(outgoing_auxinputs_index->begin() + index_entree);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_entree << ", out of range!";
        }
        if (outgoing_auxinputs->size() > index_entree) {
          (void)outgoing_auxinputs->erase(it);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_entree << ", out of range!";
        }
      } else {
        size_t idx = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
        if (outgoing_auxinputs->size() > idx) {
          outgoing_auxinputs->at(idx) = graph->nodes[node_index].node_in[0];
        } else {
          MS_LOG(DEBUG) << "Trying to index vector element at index " << idx << ", out of range!";
        }
        index_entree = LongToSize(std::distance(
          outgoing_auxinputs->begin(),
          find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), graph->nodes[node_index].node_in[0])));
      }
      for (size_t j = 1; j < graph->nodes[node_index].node_in.size(); j++) {
        exist_in_outgoing_inputs =
          find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in.begin(),
               graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end(), graph->nodes[node_index].node_in[j]);
        if (exist_in_outgoing_inputs != graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end()) {
          auto iter_j =
            find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), graph->nodes[node_index].node_in[j]);
          size_t index_remove = LongToSize(std::distance(outgoing_auxinputs->begin(), iter_j));
          if (outgoing_auxinputs_index->size() > index_remove) {
            (void)outgoing_auxinputs_index->erase(outgoing_auxinputs_index->begin() + SizeToLong(index_remove));
          } else {
            MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove << ", out of range!";
          }
          if (outgoing_auxinputs->size() > index_remove) {
            (void)outgoing_auxinputs->erase(iter_j);
          } else {
            MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove << ", out of range!";
          }
        } else {
          outgoing_auxinputs->push_back(graph->nodes[node_index].node_in[j]);
          if (outgoing_auxinputs_index->size() > index_entree) {
            outgoing_auxinputs_index->push_back(outgoing_auxinputs_index->at(index_entree));
          } else {
            MS_LOG(DEBUG) << "Trying to index vector element at index " << index_entree << ", out of range!";
          }
        }
      }
      for (size_t j = 0; j < graph->nodes[node_index].node_in_aux.size(); j++) {
        exist_in_outgoing_inputs = find(graph->nodes[graph->nodes[node_index].node_out[i]].node_in.begin(),
                                        graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end(),
                                        graph->nodes[node_index].node_in_aux[j]);
        if (exist_in_outgoing_inputs != graph->nodes[graph->nodes[node_index].node_out[i]].node_in.end()) {
          auto iter_aux_j =
            find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), graph->nodes[node_index].node_in_aux[j]);
          size_t index_remove = LongToSize(std::distance(outgoing_auxinputs->begin(), iter_aux_j));
          if (outgoing_auxinputs_index->size() > index_remove) {
            (void)outgoing_auxinputs_index->erase(outgoing_auxinputs_index->begin() + SizeToLong(index_remove));
          } else {
            MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove << ", out of range!";
          }
          if (outgoing_auxinputs->size() > index_remove) {
            (void)outgoing_auxinputs->erase(iter_aux_j);
          } else {
            MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove << ", out of range!";
          }
        } else {
          outgoing_auxinputs->push_back(graph->nodes[node_index].node_in_aux[j]);
          outgoing_auxinputs_index->push_back(outgoing_auxinputs_index->at(index_entree));
        }
      }
    } else {
      if (outgoing_auxinputs_index->size() > index_entree) {
        (void)outgoing_auxinputs_index->erase(outgoing_auxinputs_index->begin() + index_entree);
      } else {
        MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_entree << ", out of range!";
      }
      if (outgoing_auxinputs->size() > index_entree) {
        (void)outgoing_auxinputs->erase(it);
      } else {
        MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_entree << ", which is out of range.";
      }
    }
  }
}

void Eliminate_Aux_Outgoing(size_t node_index, const std::shared_ptr<Graph> &graph) {
  for (size_t i = 0; i < graph->nodes[node_index].node_out.size(); i++) {
    EliminateAuxOutgoingInput(node_index, graph, i);
    EliminateAuxOutgoingAuxInput(node_index, graph, i);
  }
}

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> &graph,
                                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                      const std::shared_ptr<std::vector<size_t>> &index_list) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t node_index = 0; node_index < graph->nodes.size(); node_index++) {
    auto type = graph->nodes[node_index].apply.op_type;
    if (ElementWiseOpType.find(type) != ElementWiseOpType.end()) {
      Eliminate_Aux(node_index, graph, eli_list);
    }
  }
  index_list->reserve(graph->nodes.size());
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    index_list->push_back(i);
  }
  for (size_t i = 0; i < eli_list->size(); i++) {
    if (eli_list->at(i)[0] >= index_list->size()) {
      MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
    }
    index_list->at(eli_list->at(i)[0]) = SIZE_MAX;
    for (size_t j = eli_list->at(i)[0] + 1; j < index_list->size(); j++) {
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
        (void)node_in->erase(node_in->cbegin() + SizeToLong(j) - 1);
      } else {
        node_in->at(j - 1) = index_list->at(node_in->at(j - 1));
      }
    }
    auto *node_in_aux = &new_graph->nodes[index_list->at(i)].node_in_aux;
    for (size_t j = node_in_aux->size(); j > 0; j--) {
      bool IsEliminated = (index_list->at(node_in_aux->at(j - 1)) == SIZE_MAX);
      if (IsEliminated) {
        (void)node_in_aux->erase(node_in_aux->begin() + SizeToLong(j) - 1);
      } else {
        node_in_aux->at(j - 1) = index_list->at(node_in_aux->at(j - 1));
      }
    }
    auto *node_out = &new_graph->nodes[index_list->at(i)].node_out;
    for (size_t j = node_out->size(); j > 0; j--) {
      bool IsEliminated = (index_list->at(node_out->at(j - 1)) == SIZE_MAX);
      if (IsEliminated) {
        (void)node_out->erase(node_out->cbegin() + SizeToLong(j) - 1);
      } else {
        node_out->at(j - 1) = index_list->at(node_out->at(j - 1));
      }
    }
  }
  return new_graph;
}
}  // namespace parallel
}  // namespace mindspore
