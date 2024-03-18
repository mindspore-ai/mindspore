/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "mindspore/core/ir/primitive.h"
#include "mindspore/core/ir/func_graph.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::parallel {
int64_t GetPrimeFactor(int64_t value) {
  static const std::vector<int64_t> prime_table = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  for (const auto &prime : prime_table) {
    if (prime > value) {
      return -1;
    }
    if (value % prime == 0) {
      return prime;
    }
  }
  return -1;
}

CNodePtr CreateShape(const AnfNodePtr &pre_cnode, const FuncGraphPtr &func_graph, const std::string &inst_name) {
  auto prim = std::make_shared<Primitive>(SHAPE_OP);
  prim->set_instance_name(inst_name);
  AnfNodePtrList shape_node_inputs(SIZE_TWO);
  shape_node_inputs[0] = NewValueNode(prim);
  shape_node_inputs[1] = pre_cnode;
  auto shape_cnode = func_graph->NewCNode(shape_node_inputs);
  return shape_cnode;
}

bool IsTargetOp(const CNodePtr &cnode, const std::string &target) {
  RETURN_IF_FALSE(cnode != nullptr);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  RETURN_IF_FALSE(value_node != nullptr);
  auto prim = value_node->value()->cast<PrimitivePtr>();
  RETURN_IF_FALSE(prim != nullptr);
  return prim->name() == target;
}

bool IsTupleGetItem(const CNodePtr &cnode) { return IsTargetOp(cnode, TUPLE_GETITEM_OP); }

bool IsReshapeOp(const CNodePtr &cnode) { return IsTargetOp(cnode, RESHAPE); }

bool IsShapeOp(const CNodePtr &cnode) { return IsTargetOp(cnode, SHAPE_OP); }

TensorRedistributionPtr GetTensorRedistributionFromCNode(const CNodePtr &node) {
  OperatorInfoPtr distribute_operator = GetDistributeOperator(node);
  if (distribute_operator == nullptr) {
    MS_LOG(WARNING) << node->fullname_with_scope() << " has no OperatorInfo.";
    return nullptr;
  }
  if (IsReshapeOp(node)) {
    return distribute_operator->reshape_tensor_redistribution();
  }
  return distribute_operator->tensor_redistribution();
}

bool IsDynamicOp(const CNodePtr &node) {
  TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(node);
  if (tensor_redistribution == nullptr) {
    return false;
  }
  return tensor_redistribution->IsAssembledStaticShape();
}

std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const std::vector<AnfNodePtr> &root_all_nodes) {
  // J->CNode->Graph
  std::set<FuncGraphPtr> graph_set;
  for (auto &node : root_all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if ((cnode->size() < 2) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_prim->name() != J && expect_prim->name() != SHARD) {
      continue;
    }
    if (IsValueNode<FuncGraph>(cnode->input(1))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_LOG(DEBUG) << "Find the forward graph success";
      (void)graph_set.insert(graph);
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graph_used = manager->func_graphs_used_total(graph);
      for (auto iter = graph_used.cbegin(); iter != graph_used.cend(); ++iter) {
        (void)graph_set.insert(*iter);
      }
    }
  }
  return graph_set;
}

AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name) {
  for (auto &param : parameters) {
    if (!ParameterIsCloned(param)) {
      continue;
    }

    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name().find(weight_name) != std::string::npos &&
        param_ptr->name().find(ACCU_GRADS) != std::string::npos) {
      MS_LOG(INFO) << "Find the accumulation grad node: " << param_ptr->name();
      return param;
    }
  }
  return nullptr;
}

std::vector<AnfNodePtr> CreateMirrorInput(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &node,
                                          const std::string &instance_name, const std::string &weight_name) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(root->manager());

  std::string op_name = op.first;
  OperatorArgs arg_forward = op.second;
  AnfNodePtr grad_accu = nullptr;

  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  if (grad_accumulation_step > 1 || split_stage_num > 1) {
    auto parameters = root->parameters();
    grad_accu = GetAccuGrad(parameters, weight_name);
    if (!grad_accu && op_name == MICRO_STEP_ALL_GATHER) {
      MS_LOG(EXCEPTION) << "You should define `accu_grads` when use " << op_name << " parameter:" << weight_name;
    }
  }

  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input;
  if (op_name == MIRROR_MINI_STEP_OPERATOR || op_name == MINI_STEP_ALL_GATHER ||
      op_name == MIRROR_MICRO_STEP_OPERATOR || op_name == MICRO_STEP_ALL_GATHER) {
    MS_EXCEPTION_IF_NULL(grad_accu);
    new_node_input = {node, grad_accu};
    MS_LOG(INFO) << "Insert the grad accumulation node as the mirror op's input";
  } else {
    new_node_input = {node};
  }

  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.cbegin() + position - 1, val);
    }
  }

  new_node_input = ConvertToRealInputs(op_name, instance_name, new_node_input, arg_forward.first);
  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

CNodePtr CreateMakeTuple(const std::vector<AnfNodePtr> &tuple_inputs, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs(tuple_inputs.size() + 1);
  make_tuple_inputs[0] = NewValueNode(prim::kPrimMakeTuple);
  for (size_t i = 0; i < tuple_inputs.size(); ++i) {
    make_tuple_inputs[i + 1] = tuple_inputs[i];
  }
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

AnfNodePtr CreateDiv(const AnfNodePtr &input_node, int64_t divisor, const FuncGraphPtr &func_graph, bool to_long,
                     const std::string &inst_name) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_ZERO("div_divisor", divisor);
  if (divisor == 1) {
    return input_node;
  }
  auto prim = std::make_shared<Primitive>(SCALAR_DIV);
  if (!inst_name.empty()) {
    prim->set_instance_name(inst_name);
  }
  std::vector<AnfNodePtr> inputs(SIZE_THREE);
  inputs[INDEX_ZERO] = NewValueNode(prim);
  inputs[INDEX_ONE] = input_node;
  inputs[INDEX_TWO] = CreatInt64Imm(divisor);
  auto div = func_graph->NewCNode(inputs);
  if (to_long) {
    auto cast_prim = NewValueNode(prim::kPrimScalarCast);
    auto type_val = MakeValue(static_cast<int64_t>(kInt64->type_id()));
    auto type_id = NewValueNode(type_val);
    auto cast = func_graph->NewCNode({cast_prim, div, type_id});
    return cast;
  }
  return div;
}

CNodePtr CreateMul(const AnfNodePtr &input_node, const int64_t factor, const FuncGraphPtr &func_graph,
                   bool to_long = false, const std::string &inst_name = "") {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_ZERO("mul_factor", factor);
  auto prim = std::make_shared<Primitive>(SCALAR_MUL);
  if (!inst_name.empty()) {
    prim->set_instance_name(inst_name);
  }
  std::vector<AnfNodePtr> inputs(SIZE_THREE);
  inputs[INDEX_ZERO] = NewValueNode(prim);
  inputs[INDEX_ONE] = input_node;
  inputs[INDEX_TWO] = CreatInt64Imm(factor);
  auto mul = func_graph->NewCNode(inputs);
  if (to_long) {
    auto cast_prim = NewValueNode(prim::kPrimScalarCast);
    auto type_val = MakeValue(static_cast<int64_t>(kInt64->type_id()));
    auto type_id = NewValueNode(type_val);
    auto cast = func_graph->NewCNode({cast_prim, mul, type_id});
    return cast;
  }
  return mul;
}

bool MatchWithPrime(const AssembledDynamicDimsMapping &dyn_dims_mapping, int64_t prime) {
  for (const auto &iter : dyn_dims_mapping) {
    int64_t prime_base = GetPrimeFactor(iter.first);
    if (prime_base == prime) {
      return true;
    }
  }
  return false;
}

bool HasDynamicDim(const Shape &shape_vec, const AssembledDynamicDimsMapping &dyn_dims_mapping,
                   const TensorRedistributionPtr &tensor_redistribution) {
  TensorLayout to_layout = tensor_redistribution->layout_transfer().to_in();
  bool is_same_rank = shape_vec.size() == to_layout.tensor_shape().array().size();
  if (!is_same_rank) {
    MS_LOG(WARNING) << "vector size is not equal, got size " + std::to_string(shape_vec.size()) + " and size " +
                         std::to_string(to_layout.tensor_shape().array().size());
  }
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    int64_t dim = shape_vec[i];
    auto iter = dyn_dims_mapping.find(dim);
    if (iter != dyn_dims_mapping.end()) {
      return true;
    }
    int64_t prime_of_dim = GetPrimeFactor(dim);
    if (prime_of_dim != -1 && MatchWithPrime(dyn_dims_mapping, prime_of_dim)) {
      return true;
    }
    if (is_same_rank) {
      int64_t full_dim = to_layout.tensor_shape().GetDimByIdx(i);
      auto ret = dyn_dims_mapping.find(full_dim);
      if (ret != dyn_dims_mapping.end()) {
        return true;
      }
    }
  }
  return false;
}

void MatchingAccordingToIndex(const Shape &shape_vec, const AssembledDynamicDimsMapping &dyn_dims_mapping,
                              const TensorRedistributionPtr &tensor_redistribution, const FuncGraphPtr &func_graph,
                              std::vector<AnfNodePtr> *shape_input) {
  MS_EXCEPTION_IF_NULL(shape_input);
  TensorLayout to_layout = tensor_redistribution->layout_transfer().to_in();
  TensorLayout from_layout = tensor_redistribution->layout_transfer().from_in();
  // If the shape not changed, it means not reshape.
  // So the dynamic dim can be matched according to index.
  // {index, {prime_dim, AnfNode}}
  std::map<size_t, std::pair<int64_t, AnfNodePtr>> mapping_table;
  for (const auto &iter : dyn_dims_mapping) {
    mapping_table.insert({iter.second.first, {iter.first, iter.second.second}});
  }
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    int64_t dim = shape_vec[i];
    if (dim != -1 && mapping_table.find(i) != mapping_table.end()) {
      std::pair<int64_t, AnfNodePtr> tuple_getitem_input_pair = mapping_table[i];
      int64_t dim_value_in_graph = tuple_getitem_input_pair.first;
      int64_t dim_prime = GetPrimeFactor(dim);
      int64_t tuple_getitem_prime = GetPrimeFactor(tuple_getitem_input_pair.first);
      if (dim_prime != tuple_getitem_prime) {
        MS_LOG(EXCEPTION) << "Prime in dim and dynamic input are not matched, " << dim_prime << " for " << dim
                          << " and " << tuple_getitem_prime << " for " << tuple_getitem_input_pair.first;
      }
      // After matching with prime, fetch the real dim value in graph and
      //  calculate whether it needs mul/div.
      if (dim_value_in_graph > dim) {
        int64_t divisor = dim_value_in_graph / dim;
        AnfNodePtr div_op =
          CreateDiv(tuple_getitem_input_pair.second, divisor, func_graph, true, "assemble_dynamic_shape_op");
        (void)shape_input->emplace_back(div_op);
        continue;
      }
      if (dim_value_in_graph < dim) {
        int64_t divisor = dim / dim_value_in_graph;
        AnfNodePtr mul_op =
          CreateMul(tuple_getitem_input_pair.second, divisor, func_graph, true, "assemble_dynamic_shape_op");
        (void)shape_input->emplace_back(mul_op);
        continue;
      }
      (void)shape_input->emplace_back(tuple_getitem_input_pair.second);
      continue;
    }
    MS_LOG(ERROR) << "Cannot find " << dim << " in shape param.";
    AnfNodePtr val = CreatInt64Imm(dim);
    (void)shape_input->emplace_back(val);
  }
}

AnfNodePtr ConvertConstParamToDynamic(const TensorRedistributionPtr &tensor_redistribution, const Param &param,
                                      const FuncGraphPtr &func_graph, bool is_reshape) {
  MS_EXCEPTION_IF_NULL(tensor_redistribution);
  AssembledDynamicDimsMapping dyn_dims_mapping = tensor_redistribution->GetDynamicDimsMapping();
  if (dyn_dims_mapping.empty()) {
    MS_LOG(ERROR) << "Doesn't have dynamic dims mapping.";
    return nullptr;
  }
  std::vector<int64_t> shape_vec = GetValue<std::vector<int64_t>>(param.first.second);
  if (shape_vec.empty()) {
    MS_LOG(ERROR) << "Cannot get shape from param.";
    return nullptr;
  }

  std::vector<AnfNodePtr> shape_input;
  if (!HasDynamicDim(shape_vec, dyn_dims_mapping, tensor_redistribution)) {
    AnfNodePtr val = NewValueNode(param.first.second);
    MS_EXCEPTION_IF_NULL(val);
    val->set_abstract(param.first.second->ToAbstract());
    return val;
  }

  MatchingAccordingToIndex(shape_vec, dyn_dims_mapping, tensor_redistribution, func_graph, &shape_input);

  if (shape_input.size() != shape_vec.size()) {
    MS_LOG(ERROR) << "shape size is not equal.";
    return nullptr;
  }
  auto make_tuple = CreateMakeTuple(shape_input, func_graph);
  return make_tuple;
}

Status ConvertStridedSliceInputs(const OperatorParams &params,
                                 const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                                 const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  for (auto &param : params) {
    if (param.first.first == BEGIN_MASK || param.first.first == END_MASK || param.first.first == ELLIPSIS_MASK ||
        param.first.first == NEW_AXIS_MASK || param.first.first == SHRINK_AXIS_MASK) {
      int64_t value = GetValue<int64_t>(param.first.second);
      MS_LOG(DEBUG) << "STRIDEDSLICE: param=" << param.first.first << ", param.second=" << value;
      AnfNodePtr val = NewValueNode(value);
      val->set_abstract(param.first.second->ToAbstract());
      (void)new_node_input->emplace_back(val);
      continue;
    }
    Shape shape_vec = GetValue<Shape>(param.first.second);
    MS_LOG(DEBUG) << "STRIDEDSLICE: param=" << param.first.first << ", " << shape_vec;
    if (param.first.first == END) {
      auto dynamic_input = ConvertConstParamToDynamic(tensor_redistribution_from_cnode, param, func_graph, false);
      MS_ERROR_IF_NULL_W_RET_VAL(dynamic_input, FAILED);
      new_node_input->emplace_back(dynamic_input);
      continue;
    }
    AnfNodePtr val = NewValueNode(shape_vec);
    MS_ERROR_IF_NULL_W_RET_VAL(val, FAILED);
    val->set_abstract(param.first.second->ToAbstract());
    (void)new_node_input->emplace_back(val);
  }
  return SUCCESS;
}

Status ConvertReshapeInputs(const OperatorParams &params,
                            const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                            const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  for (auto &param : params) {
    if (param.first.first != SHAPE) {
      continue;
    }
    Shape shape_vec = GetValue<Shape>(param.first.second);
    MS_LOG(DEBUG) << "shape param = " << shape_vec;
    auto dynamic_input = ConvertConstParamToDynamic(tensor_redistribution_from_cnode, param, func_graph, true);
    MS_ERROR_IF_NULL_W_RET_VAL(dynamic_input, FAILED);
    (void)new_node_input->emplace_back(dynamic_input);
  }
  return SUCCESS;
}

Status ConvertParamsToInputs(const Operator &op, const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                             const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  MS_ERROR_IF_NULL_W_RET_VAL(tensor_redistribution_from_cnode, FAILED);
  OperatorArgs arg_forward = op.second;
  OperatorParams params = arg_forward.second;

  if (op.first == RESHAPE) {
    if (ConvertReshapeInputs(params, tensor_redistribution_from_cnode, func_graph, new_node_input) != SUCCESS) {
      return FAILED;
    }
  } else if (op.first == STRIDEDSLICE) {
    if (ConvertStridedSliceInputs(params, tensor_redistribution_from_cnode, func_graph, new_node_input) != SUCCESS) {
      return FAILED;
    }
  } else {
    MS_LOG(DEBUG) << op.first << " is not supported.";
    return FAILED;
  }
  return SUCCESS;
}

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &pre_node, const std::string &instance_name,
                                    const CNodePtr &cur_cnode) {
  MS_EXCEPTION_IF_NULL(pre_node);
  OperatorArgs arg_forward = op.second;
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input = {pre_node};
  MS_LOG(DEBUG) << "CreateInput param.empty=" << params.empty() << ", pre_node=" << pre_node->fullname_with_scope()
                << ", op=" << op.first;
  bool is_done = false;
  if (cur_cnode != nullptr) {
    FuncGraphPtr func_graph = cur_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(cur_cnode);
    // 1. Only deal with Reshape in user scripts.
    // 2. Deal with non-user Reshape. If only have StrideSliceD, Concat and Split cannot reach.
    if (tensor_redistribution != nullptr && tensor_redistribution->IsAssembledStaticShape()) {
      MS_LOG(DEBUG) << cur_cnode->fullname_with_scope() << " distribute_operator is not nullptr";
      if (ConvertParamsToInputs(op, tensor_redistribution, func_graph, &new_node_input) == SUCCESS) {
        is_done = true;
      } else {
        MS_LOG(DEBUG) << "Convert params to inputs failed.";
      }
    } else {
      MS_LOG(DEBUG) << "cur_cnode=" << cur_cnode->fullname_with_scope() << " is not dynamic node.";
    }
  }

  if (!is_done && !params.empty()) {
    for (const auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      val->set_abstract(param.first.second->ToAbstract());
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.cbegin() + position - 1, val);
    }
  }

  new_node_input = ConvertToRealInputs(op.first, instance_name, new_node_input, arg_forward.first);

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  OperatorArgs arg_replace_op = replace_op.second;
  OperatorParams params = arg_replace_op.second;
  if (node->size() < SIZE_TWO) {
    // GetNext operator dose not has input
    if (node->size() == 1) {
      return ConvertToRealInputs(replace_op.first, instance_name, AnfNodePtrList{}, arg_replace_op.first);
    }
    MS_LOG(EXCEPTION) << "Failure: " << node->ToString() << " size is smaller than 2";
  }
  std::vector<AnfNodePtr> replace_input = {node->input(1)};

  if (replace_op.first == EMBEDDING_LOOKUP) {
    replace_input = {node->input(1), node->input(2)};
  }
  if (!params.empty() && (replace_op.first == STRIDEDSLICE || replace_op.first == RESHAPE) && IsDynamicOp(node)) {
    TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(node);
    Param param_first = *(params.begin());
    int64_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
    if (ConvertParamsToInputs(replace_op, tensor_redistribution, node->func_graph(), &replace_input) != SUCCESS) {
      MS_LOG(EXCEPTION) << "ConvertStridedSliceInputs failed.";
    }
  } else if (!params.empty()) {
    Param param_first = *(params.begin());
    int64_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      if (val == nullptr) {
        MS_LOG(EXCEPTION) << "Failure:val is nullptr";
      }
      int64_t position = param.second;
      (void)replace_input.insert(replace_input.cbegin() + position - 1, val);
    }
  } else if (replace_op.first == SYNC_BATCH_NORM) {
    for (size_t i = 2; i < node->size(); ++i) {
      replace_input.push_back(node->input(i));
    }
  }

  replace_input = ConvertToRealInputs(replace_op.first, instance_name, replace_input, arg_replace_op.first);
  SetCommunicationOpGroupLabel(replace_input);
  return replace_input;
}

void InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
                const FuncGraphPtr &func_graph, const std::string &instance_name, const std::string &param_name,
                const FuncGraphPtr &root, const TensorRedistributionPtr &tensor_redistribution) {
  // insert new node before the node
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input;

  if (root && !param_name.empty()) {
    node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  } else {
    node_input = CreateInput(op, pre_node, instance_name, node);
  }

  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_value = node_input[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node_value);
  auto new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  if (instance_name.find(NOT_RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(false));
  } else if (instance_name.find(RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(true));
  }

  auto primitive = common::AnfAlgo::GetCNodePrimitive(new_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (node->HasPrimalAttr(SEGMENT)) {
    primitive->AddAttr(SEGMENT, node->GetPrimalAttr(SEGMENT));
    new_node->AddPrimalAttr(SEGMENT, node->GetPrimalAttr(SEGMENT));
  }
  if (node->HasPrimalAttr(MICRO)) {
    new_node->AddPrimalAttr(MICRO, node->GetPrimalAttr(MICRO));
  }
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  if (instance_name.find(REDISTRIBUTION_OP) != std::string::npos) {
    new_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(new_node->UniqueId()));
    if (node->HasPrimalAttr(MICRO)) {
      new_node->AddPrimalAttr(MICRO, node->GetPrimalAttr(MICRO));
    }
  }
  manager->SetEdge(node, SizeToInt(index), new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
}

bool IsRootNode(const CNodePtr &cnode, const AnfNodePtr &root_node) {
  // cnode is TupleGetItem.
  // if first input of op is shape, and the shape first input is the same with reshape.
  // sometimes the reshape first input maybe is not same with shape first input.
  auto first_input_of_tuple_getitem = cnode->input(1)->cast<CNodePtr>();
  if (!IsTargetOp(first_input_of_tuple_getitem, SHAPE_OP)) {
    return false;
  }
  auto first_input_of_shape = first_input_of_tuple_getitem->input(1);
  if (first_input_of_shape == root_node) {
    return True;
  } else {
    MS_LOG(WARNING) << "Shape's first input is not same with root node.";
  }
  return True;
}

std::pair<CNodePtr, int64_t> FindPreviousNodeAndSkipTupleGetItem(const CNodePtr &current, int32_t depth = 0) {
  // current is TupleGetItem
  if (depth == MAX_RECURSIVE_DEPTH) {
    return {nullptr, -1};
  }
  auto prev = current->input(1);
  auto cnode = prev->cast<CNodePtr>();
  if (IsTupleGetItem(cnode)) {
    return FindPreviousNodeAndSkipTupleGetItem(cnode, depth + 1);
  }
  int64_t index = GetTupleGetItemIndex(current);
  return {cnode, index};
}

bool ModifyGraph(const CNodePtr &current_cnode, const CNodePtr &previous_tuple_getitem_cnode, size_t input_index) {
  /**
   * This function must be called after IsRootNode() called and IsRootNode() return True.
   *
   * TupleGetItem(tensor, index)
   * ->
   * ScalarMul(scalar)
   * ->
   * current_cnode
   */
  int64_t index = GetTupleGetItemIndex(previous_tuple_getitem_cnode);
  auto root_node = previous_tuple_getitem_cnode->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  if (IsTupleGetItem(root_node)) {
    // keep search the previous node.
    auto output = FindPreviousNodeAndSkipTupleGetItem(root_node);
    root_node = output.first;
  }
  // Get tensor layout from root_node.
  if (!root_node->has_user_data<OperatorInfo>()) {
    // Default/TupleGetItem-op0 has no operator info.
    MS_LOG(INFO) << root_node->fullname_with_scope() << " has no operator info.";
    return True;
  }
  OperatorInfoPtr distribute_operator = GetDistributeOperator(root_node);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  std::vector<TensorInfo> root_tensor_info = distribute_operator->outputs_tensor_info();
  if (root_tensor_info.size() != 1) {
    MS_LOG(ERROR) << "Outputs number cannot be larger than 1.";
    return False;
  }
  TensorInfo tensor_info = root_tensor_info[0];
  Map tensor_map = tensor_info.tensor_layout().tensor_map();
  Arrangement dev_arr = tensor_info.tensor_layout().device_arrangement();
  if (LongToSize(index) >= tensor_map.GetDimSize()) {
    MS_LOG(ERROR) << "Index cannot be larger than tensor_map size.";
    return False;
  }
  int64_t scalar = dev_arr.GetDimByReverseIdx(tensor_map.GetDimByIdx(index));
  // Create ValueNode for scalar->Create Mul Cnode->Modify inputs and edges
  Operator scalar_mul_op = CreateScalarMulOp(scalar);
  InsertNode(scalar_mul_op,                 // to be inserted op
             current_cnode,                 // current node
             input_index,                   // input index of current_node
             previous_tuple_getitem_cnode,  // insert scalar_mul_op between previous and current
             current_cnode->func_graph(),   // current func_graph
             "instance_name", "", nullptr);
  MS_LOG(DEBUG) << tensor_info.tensor_layout().ToString() << ", " << previous_tuple_getitem_cnode->fullname_with_scope()
                << " index: " << index << ", scalar: " << scalar;
  return True;
}

Status UpdateShapeToRootPath(const CNodePtr &cnode, const AnfNodePtr &root_node, int32_t depth = 0) {
  if (depth == MAX_RECURSIVE_DEPTH) {
    return REACH_MAX_RECURSIVE_DEPTH;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto prim = value_node->value()->cast<PrimitivePtr>();
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i)->cast<CNodePtr>();
    if (input == nullptr) {
      continue;
    }
    if (IsTupleGetItem(input) && IsRootNode(input, root_node)) {
      // Modify this graph path.
      if (!ModifyGraph(cnode, input, i)) {
        MS_LOG(ERROR) << "Failed to modify graph.";
        return Status::FAILED;
      }
      return Status::SUCCESS;
    }
    // Keep traceback.
    Status ret = UpdateShapeToRootPath(input, root_node, depth + 1);
    if (ret != Status::SUCCESS) {
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

Status UpdatePartialShape(const CNodePtr &cnode) {
  // Traceback shape_of_reshape input of Reshape Op.
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() == RESHAPE_INPUT_SIZE,
                             "Reshape op must have " + std::to_string(RESHAPE_INPUT_SIZE) + " inputs.");
  // Step1. Get second input of Reshape op which represent shape_of_reshape.
  // Step2. Visit shape_of_reshape and trace back to dynamic axis.
  auto input_of_reshape = cnode->input(RESHAPE_INPUT_SIZE - 2);
  auto shape_of_reshape = cnode->input(RESHAPE_INPUT_SIZE - 1);
  auto shape_cnode = shape_of_reshape->cast<CNodePtr>();  // MakeTuple
  if (shape_cnode == nullptr) {
    return Status::SUCCESS;
  }
  for (const auto &input : shape_cnode->inputs()) {
    auto cnode_input = input->cast<CNodePtr>();
    if (cnode_input == nullptr) {
      continue;
    }
    if (UpdateShapeToRootPath(cnode_input, input_of_reshape) != Status::SUCCESS) {
      MS_LOG(ERROR) << "Update " << cnode->fullname_with_scope() << " previous shape failed.";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

TensorInfo GetDistributeOperatorFromCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  CNodePtr target_cnode = cnode;
  if (IsTupleGetItem(cnode)) {
    // keep search the previous node.
    auto prev_node = FindPreviousNodeAndSkipTupleGetItem(cnode);
    target_cnode = prev_node.first;
  }
  if (!target_cnode->has_user_data<OperatorInfo>()) {
    MS_LOG(EXCEPTION) << target_cnode->fullname_with_scope() << " has no operator info.";
  }

  OperatorInfoPtr distribute_operator = GetDistributeOperator(target_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  std::vector<TensorInfo> root_tensor_info = distribute_operator->outputs_tensor_info();
  if (root_tensor_info.size() != 1) {
    MS_LOG(EXCEPTION) << "Outputs number cannot be larger than 1.";
  }
  return root_tensor_info[0];
}

Status UpdateShapeNode(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Step1. Get shape input tensor layout. cnode is Shape op.
  auto input_of_shape = cnode->input(1);
  auto input_cnode = input_of_shape->cast<CNodePtr>();
  if (input_cnode == nullptr) {
    return Status::SUCCESS;
  }
  TensorInfo tensor_info = GetDistributeOperatorFromCNode(input_cnode);
  Map tensor_map = tensor_info.tensor_layout().tensor_map();
  Arrangement dev_arr = tensor_info.tensor_layout().device_arrangement();

  // Step2. Get shape node users.
  auto node_users_map = func_graph->manager()->node_users();
  for (const auto &node_user : node_users_map[cnode]) {
    MS_EXCEPTION_IF_NULL(node_user.first);
    auto shape_user = node_user.first->cast<CNodePtr>();
    if (shape_user == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(IsTupleGetItem(shape_user), "Only support TupleGetItem here.");
    int64_t index = GetTupleGetItemIndex(shape_user);
    if (LongToSize(index) >= tensor_map.GetDimSize()) {
      MS_LOG(ERROR) << "Index cannot be larger than tensor_map size.";
      return Status::FAILED;
    }
    if (tensor_map.GetDimByIdx(index) < 0) {
      continue;
    }
    int64_t scalar = dev_arr.GetDimByReverseIdx(tensor_map.GetDimByIdx(index));
    for (const auto &next_node : node_users_map[shape_user]) {
      auto shape_user_user = next_node.first->cast<CNodePtr>();
      if (shape_user_user == nullptr) {
        continue;
      }
      MS_LOG(DEBUG) << shape_user->fullname_with_scope() << "->ScalarMul(" << scalar << ")->"
                    << next_node.first->fullname_with_scope() << "[" << next_node.second << "]" << std::endl;
      Operator scalar_mul_op = CreateScalarMulOp(scalar);
      InsertNode(scalar_mul_op,                  // to be inserted op
                 shape_user_user,                // current node
                 next_node.second,               // shape_user_user[input_index] = scalar_mul_op
                 shape_user,                     // insert scalar_mul_op between previous and current
                 shape_user_user->func_graph(),  // current func_graph
                 "instance_name", "", nullptr);
    }
  }
  return Status::SUCCESS;
}

Status MergeEntireShapeForDynamic(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  // Step1. Judge whether is dynamic shape.
  // Step2. Find all Shape node, get its factor arr.
  // Step3. Mul factor in Step2 to its child nodes(TupleGetItem).
  // Step4. Modify next nodes of TupleGetItem.
  auto ret_node = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_node);
  auto all_nodes = DeepScopedGraphSearch(ret_node);
  std::reverse(all_nodes.begin(), all_nodes.end());
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);

  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so mark the ops in root graph";
    auto fgs = root->manager()->func_graphs();
    for (auto fg = fgs.cbegin(); fg != fgs.cend(); ++fg) {
      // Travers all node and find shape.
      auto fg_nodes_set = (*fg)->nodes();
      for (auto const &node : fg_nodes_set) {
        if (!node->isa<CNode>()) {
          continue;
        }
        auto cnode = node->cast<CNodePtr>();
        if (!IsShapeOp(cnode)) {
          continue;
        }
        UpdateShapeNode(cnode, *fg);
      }
    }
  } else {
    MS_LOG(INFO) << "The sub graph size of root is " << root->func_graphs_used().size();
    for (auto func_graph = graph_set.cbegin(); func_graph != graph_set.cend(); ++func_graph) {
      auto return_node = (*func_graph)->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      std::vector<AnfNodePtr> all_dfs_nodes = DeepLinkedGraphSearch(return_node);
      for (const auto &node : all_dfs_nodes) {
        if (!node->isa<CNode>()) {
          continue;
        }
        auto cnode = node->cast<CNodePtr>();
        if (!IsShapeOp(cnode)) {
          continue;
        }
        UpdateShapeNode(cnode, *func_graph);
      }
    }
    for (auto const &node : all_nodes) {
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (!IsShapeOp(cnode)) {
        continue;
      }
      UpdateShapeNode(cnode, root);
    }
  }
  return Status::SUCCESS;
}
}  // namespace mindspore::parallel
