/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/graph_util/parallel_tensordump.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"

namespace mindspore::parallel {
int64_t GetPrimeFactor(int64_t value) {
  static const std::vector<int64_t> prime_table = PrimeGenerator::GetInstance()->GetPrimeTable();
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

inline bool IsTargetOp(const CNodePtr &cnode, const std::string &target) { return GetPrimName(cnode) == target; }

bool IsTupleGetItem(const CNodePtr &cnode) { return IsTargetOp(cnode, TUPLE_GETITEM_OP); }

bool IsReshapeOp(const CNodePtr &cnode) { return IsTargetOp(cnode, RESHAPE); }

bool IsShapeOp(const CNodePtr &cnode) { return IsTargetOp(cnode, SHAPE_OP); }

TensorRedistributionPtr GetTensorRedistributionFromCNode(const CNodePtr &node) {
  OperatorInfoPtr distribute_operator = GetDistributeOperator(node);
  if (distribute_operator == nullptr) {
    MS_LOG(INFO) << node->fullname_with_scope() << " has no OperatorInfo.";
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

bool IsMultiDynamicReshape(const CNodePtr &node) {
  TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(node);
  if (tensor_redistribution == nullptr) {
    return false;
  }
  return tensor_redistribution->IsMultiDynamicAxisReshape() &&
         tensor_redistribution->original_reshape_shape() != nullptr;
}

std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const std::vector<AnfNodePtr> &root_all_nodes,
                                                   const FuncGraphPtr &func_graph) {
  // J->CNode->Graph
  std::set<FuncGraphPtr> graph_set;
  bool contains_j = false;
  for (auto &node : root_all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if ((cnode->size() < SIZE_TWO) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(expect_prim);
    if (expect_prim->name() != J && expect_prim->name() != SHARD) {
      continue;
    }
    if (expect_prim->name() == J) {
      contains_j = true;
    }
    if (IsValueNode<FuncGraph>(cnode->input(1))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_LOG(DEBUG) << "Find the forward graph success";
      (void)graph_set.insert(graph);
      MS_EXCEPTION_IF_NULL(graph);
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graph_used = manager->func_graphs_used_total(graph);
      for (auto iter = graph_used.cbegin(); iter != graph_used.cend(); ++iter) {
        (void)graph_set.insert(*iter);
      }
    }
  }
  if (!contains_j && !graph_set.empty() && func_graph != nullptr) {
    (void)graph_set.insert(func_graph);
  }
  return graph_set;
}

AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name) {
  for (auto &param : parameters) {
    MS_EXCEPTION_IF_NULL(param);
    if (!ParameterIsCloned(param)) {
      continue;
    }

    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    auto accu_grads_name = std::string(ACCU_GRADS) + "." + weight_name;
    if (param_ptr->name() == accu_grads_name) {
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
      MS_LOG_WITH_NODE(EXCEPTION, node) << "You should define `accu_grads` when use " << op_name
                                        << " parameter:" << weight_name;
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

CNodePtr CreateMakeTuple(const std::vector<AnfNodePtr> &tuple_inputs, const FuncGraphPtr &func_graph,
                         const std::string &instance_name = "") {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs(tuple_inputs.size() + 1);
  auto prim = std::make_shared<Primitive>(MAKE_TUPLE);
  if (!instance_name.empty()) {
    prim->set_instance_name(instance_name);
  }
  make_tuple_inputs[0] = NewValueNode(prim);
  for (size_t i = 0; i < tuple_inputs.size(); ++i) {
    make_tuple_inputs[i + 1] = tuple_inputs[i];
  }
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

CNodePtr CreateSplit(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph,
                     const std::string &inst_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == SIZE_THREE, "inputs is empty.");
  auto prim = std::make_shared<Primitive>(SPLIT);
  if (!inst_name.empty()) {
    prim->set_instance_name(inst_name);
  }
  std::vector<AnfNodePtr> split_inputs(SIZE_FOUR);
  split_inputs[INDEX_ZERO] = NewValueNode(prim);
  split_inputs[INDEX_ONE] = inputs[INDEX_ZERO];   // split_input
  split_inputs[INDEX_TWO] = inputs[INDEX_ONE];    // split_axis
  split_inputs[INDEX_THREE] = inputs[INDEX_TWO];  // split_size
  auto split = func_graph->NewCNode(split_inputs);
  return split;
}

AnfNodePtr CreateDiv(const AnfNodePtr &input_node, int64_t divisor, const FuncGraphPtr &func_graph,
                     const std::string &inst_name) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_ZERO("div_divisor", divisor);
  if (divisor == 1) {
    return input_node;
  }
  auto prim = std::make_shared<Primitive>(SCALAR_FLOOR_DIV);
  if (!inst_name.empty()) {
    prim->set_instance_name(inst_name);
  }
  std::vector<AnfNodePtr> inputs(SIZE_THREE);
  inputs[INDEX_ZERO] = NewValueNode(prim);
  inputs[INDEX_ONE] = input_node;
  inputs[INDEX_TWO] = CreatInt64Imm(divisor);
  auto div = func_graph->NewCNode(inputs);
  return div;
}

CNodePtr CreateMul(const AnfNodePtr &input_node, const int64_t factor, const FuncGraphPtr &func_graph,
                   const std::string &inst_name = "") {
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

inline bool IsSameRank(const Shape &shape_vec, const Shape &targe_shape_vec) {
  return shape_vec.size() == targe_shape_vec.size();
}

bool HasAssembledDynamicDim(const Shape &shape_vec, const AssembledDynamicDimsMapping &dyn_dims_mapping,
                            const TensorRedistributionPtr &tensor_redistribution, bool is_same_rank) {
  for (int64_t dim : shape_vec) {
    auto iter = dyn_dims_mapping.find(dim);
    if (iter != dyn_dims_mapping.end()) {
      return true;
    }
    int64_t prime_base = dim;
    while (prime_base > 1) {
      int64_t prime_of_dim = GetPrimeFactor(prime_base);
      if (prime_of_dim == -1) {
        break;
      }
      if (MatchWithPrime(dyn_dims_mapping, prime_of_dim)) {
        return true;
      }
      prime_base /= prime_of_dim;
    }
  }
  return false;
}

void MatchingAccordingToPrime(const Shape &shape_vec, const AssembledDynamicDimsMapping &dyn_dims_mapping,
                              const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *shape_input,
                              enum ReshapeMode reshape_mode = ReshapeMode::NO_RESHAPE) {
  MS_LOG(INFO) << "Match with prime, shape_vec=" << shape_vec << ", reshape_mode=" << reshape_mode;
  MS_EXCEPTION_IF_NULL(shape_input);
  // If the shape not changed, it means not reshape.
  // So the dynamic dim can be matched according to index.
  std::string instance_name = std::string(REDISTRIBUTION_OP) + "_" + "assemble_shape";
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    int64_t dim = shape_vec[i];
    // TODO(liuchongming): dim could has more than one prime, have to get all prime in dim.
    int64_t dim_prime = GetPrimeFactor(dim);
    bool found = false;
    if (dim != -1 && dim_prime != -1) {
      for (const auto &iter : dyn_dims_mapping) {
        int64_t dim_value_in_graph = iter.first;
        AnfNodePtr tuple_getitem = iter.second.second;
        int64_t dyn_prime = GetPrimeFactor(dim_value_in_graph);
        if (dyn_prime != dim_prime) {
          continue;
        }
        MS_LOG(INFO) << "i=" << i << ", dim_value_in_graph=" << dim_value_in_graph << ", dim_prime=" << dim_prime
                     << ", dim=" << dim;
        if (dim_value_in_graph > dim) {
          int64_t divisor = dim_value_in_graph / dim;
          AnfNodePtr div_op = CreateDiv(tuple_getitem, divisor, func_graph, instance_name);
          (void)shape_input->emplace_back(div_op);
          found = true;
          break;
        } else if (dim_value_in_graph < dim) {
          int64_t divisor = dim / dim_value_in_graph;
          AnfNodePtr mul_op = CreateMul(tuple_getitem, divisor, func_graph, instance_name);
          (void)shape_input->emplace_back(mul_op);
          found = true;
          break;
        } else {
          (void)shape_input->emplace_back(tuple_getitem);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      MS_LOG(INFO) << "Cannot find " << dim << " in shape param.";
      AnfNodePtr val = CreatInt64Imm(dim);
      (void)shape_input->emplace_back(val);
    }
  }
}

void MatchingAccordingToIndex(const Shape &shape_vec, const AssembledDynamicDimsMapping &dyn_dims_mapping,
                              const TensorRedistributionPtr &tensor_redistribution, const FuncGraphPtr &func_graph,
                              std::vector<AnfNodePtr> *shape_input,
                              enum ReshapeMode reshape_mode = ReshapeMode::NO_RESHAPE) {
  MS_LOG(INFO) << "Match with index, shape_vec=" << shape_vec;
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
          CreateDiv(tuple_getitem_input_pair.second, divisor, func_graph, "assemble_dynamic_shape_op");
        (void)shape_input->emplace_back(div_op);
        continue;
      }
      if (dim_value_in_graph < dim) {
        int64_t divisor = dim / dim_value_in_graph;
        AnfNodePtr mul_op =
          CreateMul(tuple_getitem_input_pair.second, divisor, func_graph, "assemble_dynamic_shape_op");
        (void)shape_input->emplace_back(mul_op);
        continue;
      }
      (void)shape_input->emplace_back(tuple_getitem_input_pair.second);
      continue;
    }
    MS_LOG(INFO) << "Cannot find " << dim << " in shape param.";
    AnfNodePtr val = CreatInt64Imm(dim);
    (void)shape_input->emplace_back(val);
  }
}

int64_t CountDynamicAxis(const AnfNodePtrList &shape_input) {
  int64_t dyn_axis_cnt = 0;
  for (size_t i = 0; i < shape_input.size(); ++i) {
    if (shape_input[i]->isa<ValueNode>()) {
      auto val_node = shape_input[i]->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(val_node);
      MS_EXCEPTION_IF_NULL(val_node->value());
      int64_t index = GetValue<int64_t>(val_node->value());
      if (index == -1) {
        dyn_axis_cnt += 1;
      }
    } else {
      dyn_axis_cnt += 1;
    }
  }
  return dyn_axis_cnt;
}

inline bool WhetherIsValueNode(const AnfNodePtr &node) { return node->isa<ValueNode>(); }

AnfNodePtr ConvertConstParamToDynamic(const TensorRedistributionPtr &tensor_redistribution, const Param &param,
                                      const FuncGraphPtr &func_graph, bool is_reshape,
                                      enum ReshapeMode reshape_mode = ReshapeMode::NO_RESHAPE) {
  // Only ConvertReshapeInputs will use this function.
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

  // After refactor, dyn_dims_mapping is generated according to origin_from_shape.
  // Reshape has 3 scenes:
  // 1. from_origin_->from_layout.from: when shape is squeezed, 1 in front or in back are removed from from_origin.
  // 2. to_layout.to->to_origin_: when shape is unified, it could be expanded.
  // 3. User's reshape: written in user's scripts.
  Shape origin_from_shape = tensor_redistribution->from_origin_layout().tensor_shape().array();
  Shape origin_slice_from_shape = tensor_redistribution->from_origin_layout().slice_shape().array();
  Shape from_shape = tensor_redistribution->from_layout().tensor_shape().array();
  Shape unified_from_shape = tensor_redistribution->layout_transfer().from_in().tensor_shape().array();
  Shape unified_slice_from_shape = tensor_redistribution->layout_transfer().from_in().slice_shape().array();
  MS_LOG(INFO) << "reshape_mode=" << reshape_mode << ", shape_vec: " << shape_vec
               << ", origin_from_shape: " << origin_from_shape
               << ", \norigin_slice_from_shape: " << origin_slice_from_shape << ", \nfrom_shape: " << from_shape
               << ", \nunified_from_shape: " << unified_from_shape
               << ", \nunified_slice_from_shape:" << unified_slice_from_shape;
  // The rank should be compared between shape_vec and origin_from_shape, because
  // the mapping is generated according to origin_from_shape.
  bool is_same_rank = IsSameRank(shape_vec, origin_from_shape);
  if (!HasAssembledDynamicDim(shape_vec, dyn_dims_mapping, tensor_redistribution, is_same_rank)) {
    // If the shape_vec is (-1, dim_1) and dim_1 is not a generated fake value by tensor redistribution,
    // so it doesn't have to match.
    AnfNodePtr val = NewValueNode(param.first.second);
    MS_EXCEPTION_IF_NULL(val);
    return val;
  }
  if (shape_vec.size() == 1) {
    std::vector<int64_t> const_shape{-1};
    AnfNodePtr val = NewValueNode(const_shape);
    return val;
  }
  std::vector<AnfNodePtr> shape_input;
  if (reshape_mode == ReshapeMode::FROM_ORIGIN_SLICE_TO_FROM_LAYOUT_SLICE ||
      reshape_mode == ReshapeMode::TO_ORIGIN_SLICE_TO_TO_LAYOUT_SLICE ||
      reshape_mode == ReshapeMode::FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE) {
    MatchingAccordingToPrime(shape_vec, dyn_dims_mapping, func_graph, &shape_input, reshape_mode);
  } else {
    if (is_same_rank && reshape_mode != ReshapeMode::FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE) {
      MatchingAccordingToIndex(shape_vec, dyn_dims_mapping, tensor_redistribution, func_graph, &shape_input,
                               reshape_mode);
    } else {
      MatchingAccordingToPrime(shape_vec, dyn_dims_mapping, func_graph, &shape_input, reshape_mode);
    }
  }
  if (shape_input.size() != shape_vec.size()) {
    MS_LOG(ERROR) << "shape size is not equal.";
    return nullptr;
  }

  if (is_reshape) {
    // If only has one dynamic axis, then set it to -1.
    size_t dyn_axis_cnt = LongToSize(CountDynamicAxis(shape_input));
    MS_LOG(INFO) << "For shape_vec=" << shape_vec << ", has " << dyn_axis_cnt << " dynamic axis.";
    if (dyn_axis_cnt == 1) {
      constexpr int64_t unknown = -1;
      for (size_t i = 0; i < shape_input.size(); ++i) {
        if (shape_input[i]->isa<CNode>()) {
          shape_input[i] = NewValueNode(MakeValue(unknown));
          MS_LOG(INFO) << "change index " << i << " to -1.";
          break;
        }
      }
    }
  }
  if (std::all_of(shape_input.begin(), shape_input.end(), &WhetherIsValueNode)) {
    std::vector<int64_t> const_shape(shape_input.size());
    for (size_t i = 0; i < shape_input.size(); ++i) {
      auto val_node = shape_input[i]->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(val_node);
      MS_EXCEPTION_IF_NULL(val_node->value());
      int64_t value = GetValue<int64_t>(val_node->value());
      const_shape[i] = value;
    }
    return NewValueNode(const_shape);
  }
  auto make_tuple = CreateMakeTuple(shape_input, func_graph, REDISTRIBUTION_OP);
  return make_tuple;
}

Status ConvertStridedSliceInputs(const OperatorParams &params,
                                 const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                                 const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  for (auto &param : params) {
    if (param.first.first == BEGIN_MASK || param.first.first == END_MASK || param.first.first == ELLIPSIS_MASK ||
        param.first.first == NEW_AXIS_MASK || param.first.first == SHRINK_AXIS_MASK) {
      int64_t value = GetValue<int64_t>(param.first.second);
      MS_LOG(INFO) << "STRIDEDSLICE: param=" << param.first.first << ", param.second=" << value;
      AnfNodePtr val = NewValueNode(value);
      val->set_abstract(param.first.second->ToAbstract());
      (void)new_node_input->emplace_back(val);
      continue;
    }
    Shape shape_vec = GetValue<Shape>(param.first.second);
    MS_LOG(INFO) << "STRIDEDSLICE: param=" << param.first.first << ", " << shape_vec;
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

bool WhetherMatchingIsNeededForReshape(const Shape &shape_vec, const TensorRedistributionPtr &tensor_redistribution) {
  size_t user_specific_dynamic_dim_cnt = std::count(shape_vec.begin(), shape_vec.end(), -1);
  TensorLayout to_layout = tensor_redistribution->layout_transfer().to_in();
  Shape to_shape_in_layout = to_layout.slice_shape().array();
  MS_LOG(INFO) << "shape_vec=" << shape_vec << ", to_shape_in_layout=" << to_shape_in_layout;
  if (user_specific_dynamic_dim_cnt == 1 && shape_vec.size() == to_shape_in_layout.size()) {
    size_t dyn_index = static_cast<size_t>(std::find(shape_vec.begin(), shape_vec.end(), -1) - shape_vec.begin());
    for (size_t i = 0; i < shape_vec.size(); ++i) {
      if (i != dyn_index && shape_vec[i] != to_shape_in_layout[i]) {
        return true;
      }
    }
    MS_LOG(INFO) << "No need to matching for shape: " << shape_vec << ", to_shape_in_layout: " << to_shape_in_layout;
    return false;
  }
  return true;
}

inline bool HasOnlyOneDynamicAxis(const Shape &shape_vec,
                                  const TensorRedistributionPtr &tensor_redistribution_from_cnode) {
  Shape origin_to_no_assembled = tensor_redistribution_from_cnode->to_origin_no_assembled().tensor_shape().array();
  Shape origin_to_no_assembled_slice = tensor_redistribution_from_cnode->to_origin_no_assembled().slice_shape().array();
  bool has_only_one_dynamic_axis = std::count(origin_to_no_assembled.begin(), origin_to_no_assembled.end(), -1) == 1;
  MS_LOG(INFO) << "shape_vec: " << shape_vec << ", origin_to_no_assembled: " << origin_to_no_assembled
               << ", origin_to_no_assembled_slice: " << origin_to_no_assembled_slice;
  return (origin_to_no_assembled.size() == shape_vec.size()) && has_only_one_dynamic_axis;
}

void ReplaceDynamicAxisToNegOne(const TensorRedistributionPtr &tensor_redistribution_from_cnode, Shape *shape_vec) {
  Shape origin_to_no_assembled = tensor_redistribution_from_cnode->to_origin_no_assembled().tensor_shape().array();
  for (size_t i = 0; i < origin_to_no_assembled.size(); ++i) {
    if (origin_to_no_assembled[i] == -1) {
      (*shape_vec)[i] = -1;
    }
  }
}

bool SkipForOneDynamicAxisInTargetShape(const Shape &shape_vec,
                                        const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                                        const enum ReshapeMode &mode) {
  if (mode == ReshapeMode::FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE) {
    auto to_origin_no_assembled = tensor_redistribution_from_cnode->to_origin_no_assembled();
    Shape origin_to_no_assembled_base_slice = to_origin_no_assembled.base_slice_shape().array();
    bool has_one_dynamic_axis =
      std::count(origin_to_no_assembled_base_slice.begin(), origin_to_no_assembled_base_slice.end(), -1) == 1;
    if (has_one_dynamic_axis) {
      return true;
    }
  }
  return false;
}

bool UpdateOneDynamicShapeAxisToNegOne(Shape *shape_vec,
                                       const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                                       const enum ReshapeMode &mode) {
  if (mode == ReshapeMode::FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE) {
    auto to_origin_no_assembled = tensor_redistribution_from_cnode->to_origin_no_assembled();
    Shape origin_to_no_assembled_base_slice = to_origin_no_assembled.base_slice_shape().array();
    for (size_t i = 0; i < origin_to_no_assembled_base_slice.size(); ++i) {
      if (origin_to_no_assembled_base_slice[i] == -1) {
        (*shape_vec)[i] = -1;
        return true;
      }
    }
    return false;
  }
  MS_LOG(EXCEPTION) << "Unsupported reshape mode: " << mode;
}

Status ConvertReshapeInputs(const OperatorParams &params,
                            const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                            const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  MS_EXCEPTION_IF_NULL(tensor_redistribution_from_cnode);
  Param shape_param;
  bool use_origin_shape = false;
  ReshapeMode reshape_mode = ReshapeMode::NO_RESHAPE;
  for (auto &param : params) {
    if (param.first.first == SHAPE) {
      shape_param = param;
      continue;
    }
    if (param.first.first == USE_ORIGIN_SHAPE) {
      use_origin_shape = GetValue<bool>(param.first.second);
      MS_LOG(INFO) << "Has USE_ORIGIN_SHAPE = " << use_origin_shape;
      continue;
    }
    if (param.first.first == REDISTRIBUTION_RESHAPE_MODE) {
      reshape_mode = static_cast<ReshapeMode>(GetValue<int64_t>(param.first.second));
      MS_LOG(INFO) << "Has REDISTRIBUTION_RESHAPE_MODE = " << reshape_mode;
      continue;
    }
  }
  Shape shape_vec = GetValue<Shape>(shape_param.first.second);
  if (shape_vec.size() == 1) {
    std::vector<int64_t> const_shape{-1};
    AnfNodePtr val = NewValueNode(const_shape);
    (void)new_node_input->emplace_back(val);
    return SUCCESS;
  }
  if ((use_origin_shape || tensor_redistribution_from_cnode->IsMultiDynamicAxisReshape()) &&
      tensor_redistribution_from_cnode->original_reshape_shape() != nullptr) {
    // Only reshape in user's code should be in this branch.
    // original_reshape_shape could be ValueNode, MakeTuple, Shape.
    (void)new_node_input->emplace_back(tensor_redistribution_from_cnode->original_reshape_shape());
    return SUCCESS;
  }
  size_t dynamic_axis_cnt = std::count(shape_vec.begin(), shape_vec.end(), -1);
  if (shape_vec.size() > 1 && dynamic_axis_cnt >= SIZE_TWO) {
    MS_LOG(WARNING) << "The shape of Reshape op has more than one -1, cannot be supported for now.";
  }
  Shape origin_to_no_assembled = tensor_redistribution_from_cnode->to_origin_no_assembled().tensor_shape().array();
  Shape origin_to_no_assembled_slice = tensor_redistribution_from_cnode->to_origin_no_assembled().slice_shape().array();
  MS_LOG(INFO) << "shape_vec: " << shape_vec << ", reshape_mode: " << reshape_mode
               << ", origin_to_no_assembled: " << origin_to_no_assembled
               << ", origin_to_no_assembled_slice: " << origin_to_no_assembled_slice;
  // if only has one dynamic axis, then replace it with -1 simply.
  if ((reshape_mode == ReshapeMode::NO_RESHAPE || reshape_mode == ReshapeMode::FROM_ORIGIN_SLICE_TO_FROM_LAYOUT_SLICE ||
       reshape_mode == ReshapeMode::TO_ORIGIN_SLICE_TO_TO_LAYOUT_SLICE) &&
      HasOnlyOneDynamicAxis(shape_vec, tensor_redistribution_from_cnode)) {
    // After HasOnlyOneDynamicAxis checks, shape_vec must have one dynamic axis and it must be prime axis.
    Shape new_shape_vec(shape_vec);
    ReplaceDynamicAxisToNegOne(tensor_redistribution_from_cnode, &new_shape_vec);
    MS_LOG(INFO) << "Replace shape: " << shape_vec << " to new_shape_vec: " << new_shape_vec;
    AnfNodePtr val = NewValueNode(new_shape_vec);
    (void)new_node_input->emplace_back(val);
    return SUCCESS;
  }
  if (SkipForOneDynamicAxisInTargetShape(shape_vec, tensor_redistribution_from_cnode, reshape_mode)) {
    Shape new_shape_vec(shape_vec);
    UpdateOneDynamicShapeAxisToNegOne(&new_shape_vec, tensor_redistribution_from_cnode, reshape_mode);
    AnfNodePtr val = NewValueNode(new_shape_vec);
    (void)new_node_input->emplace_back(val);
    return SUCCESS;
  }
  if (!WhetherMatchingIsNeededForReshape(shape_vec, tensor_redistribution_from_cnode)) {
    MS_LOG(INFO) << "No need to matching for " << shape_vec;
    AnfNodePtr val = NewValueNode(shape_param.first.second);
    (void)new_node_input->emplace_back(val);
    return SUCCESS;
  }
  auto dynamic_input =
    ConvertConstParamToDynamic(tensor_redistribution_from_cnode, shape_param, func_graph, true, reshape_mode);
  MS_ERROR_IF_NULL_W_RET_VAL(dynamic_input, FAILED);
  (void)new_node_input->emplace_back(dynamic_input);
  return SUCCESS;
}

Status ConvertSplitInputs(const OperatorParams &params, const FuncGraphPtr &func_graph,
                          std::vector<AnfNodePtr> *new_node_input) {
  MS_EXCEPTION_IF_CHECK_FAIL(new_node_input->size() == 1,
                             "new_node_input must and only contain the input of split for split.");
  auto split_target = new_node_input[0];
  std::vector<AnfNodePtr> split_inputs = {split_target};
  ValuePtr output_index;
  for (auto &param : params) {
    if (param.first.first == SPLIT_OUTPUT_INDEX) {
      output_index = param.first.second;
      continue;
    }
    AnfNodePtr val = NewValueNode(param.first.second);
    MS_EXCEPTION_IF_NULL(val);
    (void)split_inputs.emplace_back(val);
  }
  constexpr char tag[] = "redistribution_allsplit";
  auto split_op = CreateSplit(split_inputs, func_graph, tag);
  auto split_output_index = NewValueNode(output_index);
  auto tuple_get_item_prim = std::make_shared<Primitive>(TUPLE_GETITEM_OP);
  auto prim_value_node = NewValueNode(tuple_get_item_prim);
  tuple_get_item_prim->set_instance_name(tag);
  new_node_input->resize(SIZE_THREE);
  (*new_node_input)[INDEX_ZERO] = prim_value_node;
  (*new_node_input)[INDEX_ONE] = split_op;
  (*new_node_input)[INDEX_TWO] = split_output_index;
  return SUCCESS;
}

bool IsToBeInsertedSplitOp(const Operator &op) {
  // if split op has attr SPLIT_INSERT_LATER, then skip it in OptimizeTensorRedistributionOperatorList stage,
  // and insert it in CreateInputs
  if (op.first != SPLIT) {
    return false;
  }
  OperatorAttrs op_attrs = op.second.first;
  auto is_skip_func = [](const Attr &attr) -> bool {
    return attr.first == SPLIT_INSERT_LATER && GetValue<bool>(attr.second);
  };
  return std::any_of(op_attrs.begin(), op_attrs.end(), is_skip_func);
}

Status ConvertParamsToInputs(const Operator &op, const TensorRedistributionPtr &tensor_redistribution_from_cnode,
                             const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *new_node_input) {
  MS_ERROR_IF_NULL_W_RET_VAL(tensor_redistribution_from_cnode, FAILED);
  MS_EXCEPTION_IF_NULL(func_graph);
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
  } else if (IsToBeInsertedSplitOp(op)) {
    if (ConvertSplitInputs(params, func_graph, new_node_input) != SUCCESS) {
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
  MS_LOG(INFO) << "CreateInput param.empty=" << params.empty() << ", pre_node=" << pre_node->fullname_with_scope()
               << ", op=" << op.first;
  bool is_done = false;
  if (cur_cnode != nullptr && (IsDynamicOp(cur_cnode) || IsMultiDynamicReshape(cur_cnode))) {
    TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(cur_cnode);
    // 1. Only deal with Reshape in user scripts.
    // 2. Deal with non-user Reshape. If only have StrideSliceD, Concat and Split cannot reach.
    MS_LOG(DEBUG) << cur_cnode->fullname_with_scope() << " distribute_operator is not nullptr";
    if (ConvertParamsToInputs(op, tensor_redistribution, cur_cnode->func_graph(), &new_node_input) == SUCCESS) {
      is_done = true;
    } else {
      MS_LOG(INFO) << "Convert params to inputs failed.";
    }
  }

  if (IsToBeInsertedSplitOp(op) && !is_done && cur_cnode != nullptr) {
    // it means Split on static shape scene.
    auto ret = ConvertSplitInputs(params, cur_cnode->func_graph(), &new_node_input);
    MS_EXCEPTION_IF_CHECK_FAIL(ret == SUCCESS, "Insert split op failed.");
    is_done = true;
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

  if (!IsToBeInsertedSplitOp(op)) {
    new_node_input = ConvertToRealInputs(op.first, instance_name, new_node_input, arg_forward.first);
  }
  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());
  OperatorArgs arg_replace_op = replace_op.second;
  OperatorParams params = arg_replace_op.second;
  if (node->size() < SIZE_TWO) {
    // GetNext operator does not has input
    if (node->size() == 1) {
      return ConvertToRealInputs(replace_op.first, instance_name, AnfNodePtrList{}, arg_replace_op.first);
    }
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure: " << node->ToString() << " size is smaller than 2";
  }
  std::vector<AnfNodePtr> replace_input = {node->input(1)};

  if (replace_op.first == EMBEDDING_LOOKUP) {
    replace_input = {node->input(1), node->input(2)};
  }
  if (!params.empty() && replace_op.first != SYNC_BATCH_NORM) {
    Param param_first = *(params.begin());
    int64_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
  }
  bool is_done = false;
  bool to_be_converted = replace_op.first == SPLIT || replace_op.first == STRIDEDSLICE || replace_op.first == RESHAPE;
  if (!params.empty() && to_be_converted && (IsDynamicOp(node) || IsMultiDynamicReshape(node))) {
    TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(node);
    auto ret = ConvertParamsToInputs(replace_op, tensor_redistribution, node->func_graph(), &replace_input);
    MS_EXCEPTION_IF_CHECK_FAIL(ret == SUCCESS, "ConvertStridedSliceInputs failed.");
    is_done = true;
  } else if (!params.empty() && !IsToBeInsertedSplitOp(replace_op)) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      if (val == nullptr) {
        MS_LOG_WITH_NODE(EXCEPTION, val) << "Failure:val is nullptr";
      }
      int64_t position = param.second;
      (void)replace_input.insert(replace_input.cbegin() + position - 1, val);
    }
  } else if (replace_op.first == SYNC_BATCH_NORM) {
    for (size_t i = 2; i < node->size(); ++i) {
      replace_input.push_back(node->input(i));
    }
  }

  if (!IsToBeInsertedSplitOp(replace_op)) {
    replace_input = ConvertToRealInputs(replace_op.first, instance_name, replace_input, arg_replace_op.first);
  } else if (IsToBeInsertedSplitOp(replace_op) && !is_done) {
    // it means Split on static shape scene.
    auto ret = ConvertSplitInputs(params, node->func_graph(), &replace_input);
    MS_EXCEPTION_IF_CHECK_FAIL(ret == SUCCESS, "Insert split op failed.");
  }
  SetCommunicationOpGroupLabel(replace_input);
  return replace_input;
}

CNodePtr InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
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
  MS_EXCEPTION_IF_NULL(new_node_prim);
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
  return new_node;
}

CNodePtr FindPreviousCareNode(const CNodePtr &current, int32_t depth = 0) {
  if (depth == MAX_RECURSIVE_DEPTH) {
    return nullptr;
  }
  size_t travel_index = IsSomePrimitive(current, DUMPGRADIENT) ? kDumpGradientSkipIndex : kIndex1;
  auto prev = current->input(travel_index);
  // If prev is parameter maybe problem here.
  auto cnode = prev->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(INFO) << "Input of node is not a cnode: " << prev->fullname_with_scope();
    return nullptr;
  }
  static std::set<std::string> RECURSIVE_WHITE_LIST = {CAST, TUPLE_GETITEM_OP, INSERTGRADIENTOF, DEPEND, DUMPGRADIENT};
  if (!IsParallelCareNode(cnode) && (RECURSIVE_WHITE_LIST.find(GetPrimName(cnode)) != RECURSIVE_WHITE_LIST.end())) {
    return FindPreviousCareNode(cnode, depth + 1);
  }
  return cnode;
}

Status GetDistributeOperatorFromCNode(const CNodePtr &cnode, TensorInfo *tensor_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  CNodePtr target_cnode = cnode;
  if (!IsParallelCareNode(cnode)) {
    // keep search the previous node.
    target_cnode = FindPreviousCareNode(cnode);
  }
  if (target_cnode == nullptr) {
    return Status::FAILED;
  }
  if (!target_cnode->has_user_data<OperatorInfo>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Found " << cnode->fullname_with_scope() << " previous node is "
                                       << target_cnode->fullname_with_scope() << " and it has no operator info.";
  }

  OperatorInfoPtr distribute_operator = GetDistributeOperator(target_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  std::vector<TensorInfo> root_tensor_info = distribute_operator->outputs_tensor_info();
  if (root_tensor_info.size() != 1) {
    if (IsTupleGetItem(cnode)) {
      int64_t output_index = GetTupleGetItemIndex(cnode);
      MS_EXCEPTION_IF_CHECK_FAIL(
        (output_index >= 0 && output_index < SizeToLong(root_tensor_info.size())),
        "TupleGetItem index is not matched with its input length, TupleGetItem is " + cnode->fullname_with_scope());
      MS_LOG(INFO) << "Replace tensor info use " << target_cnode->fullname_with_scope() << " with index "
                   << output_index;
      (*tensor_info) = root_tensor_info[output_index];
      return Status::SUCCESS;
    }
    MS_LOG(INFO) << "Outputs number cannot be larger than 1, but " << target_cnode->fullname_with_scope() << " has "
                 << root_tensor_info.size() << " outputs.";
  }
  (*tensor_info) = root_tensor_info[0];
  return Status::SUCCESS;
}

Status UpdateTupleGetItemShapeValue(const CNodePtr &tuple_getitem, const TensorInfo &tensor_info,
                                    const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "into UpdateTupleGetItemShapeValue";
  Map tensor_map = tensor_info.tensor_layout().tensor_map();
  Arrangement dev_arr = tensor_info.tensor_layout().device_arrangement();
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users_map = manager->node_users();

  int64_t index = GetTupleGetItemIndex(tuple_getitem);
  if (LongToSize(index) >= tensor_map.GetDimSize()) {
    MS_LOG(ERROR) << "Index cannot be larger than tensor_map size.";
    return Status::FAILED;
  }
  if (tensor_map.GetDimByIdx(index) < 0) {
    MS_LOG(DEBUG) << "Skip index " << index << ", because it's " << tensor_map.GetDimByIdx(index);
    return Status::SUCCESS;
  }
  int64_t scalar = dev_arr.GetDimByReverseIdx(tensor_map.GetDimByIdx(index));
  for (const auto &next_node : node_users_map[tuple_getitem]) {
    auto tuple_getitem_user = next_node.first->cast<CNodePtr>();
    if (tuple_getitem_user == nullptr) {
      MS_LOG(DEBUG) << "tuple_getitem_user is nullptr";
      continue;
    }
    MS_LOG(INFO) << tuple_getitem->input(1)->fullname_with_scope() << "->" << tuple_getitem->fullname_with_scope()
                 << "->ScalarMul(" << scalar << ")->" << next_node.first->fullname_with_scope() << "["
                 << next_node.second << "]" << std::endl;
    Operator scalar_mul_op = CreateScalarMulOp(scalar);
    (void)InsertNode(scalar_mul_op,                     // to be inserted op
                     tuple_getitem_user,                // current node
                     next_node.second,                  // tuple_getitem_user[input_index] = scalar_mul_op
                     tuple_getitem,                     // insert scalar_mul_op between previous and current
                     tuple_getitem_user->func_graph(),  // current func_graph
                     "update_partial_shape", "", nullptr);
  }
  return Status::SUCCESS;
}

Status UpdateReshapeShapeValue(const CNodePtr &reshape_cnode, const CNodePtr &shape_cnode, const Shape &shape,
                               const TensorInfo &tensor_info, const FuncGraphPtr &func_graph) {
  // Replace shape to MakeTuple(shape[0]*factor0, shape[1]*factor1,...)
  MS_LOG(INFO) << "into UpdateReshapeShapeValue: " << shape;
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  MS_EXCEPTION_IF_NULL(shape_cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  Map tensor_map = tensor_info.tensor_layout().tensor_map();
  Arrangement dev_arr = tensor_info.tensor_layout().device_arrangement();
  TensorRedistributionPtr tensor_redistribution = GetTensorRedistributionFromCNode(reshape_cnode);
  MS_LOG(INFO) << "shape: " << shape << ", tensor_map: " << tensor_map.array() << ", dev_arr: " << dev_arr.array();
  std::vector<AnfNodePtr> make_tuple_inputs;
  std::string instance_name = std::string(REDISTRIBUTION_OP) + "_replace_reshape";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 0) {
      ValueNodePtr const_val_node = nullptr;
      if (tensor_map.array().size() == shape.size() && tensor_map.GetDimByIdx(i) != -1) {
        int64_t scalar = dev_arr.GetDimByReverseIdx(tensor_map.GetDimByIdx(i));
        MS_EXCEPTION_IF_CHECK_FAIL(shape[i] % scalar == 0,
                                   "Const shape cannot be divided for index " + std::to_string(i));
        int64_t partial_shape = shape[i] / scalar;
        const_val_node = NewValueNode(MakeValue(partial_shape));
      } else {
        // Get const value and set to make_tuple_inputs.
        const_val_node = NewValueNode(MakeValue(shape[i]));
      }
      make_tuple_inputs.emplace_back(const_val_node);
      MS_LOG(INFO) << "Create ValueNode " << shape[i];
      continue;
    }
    // Get shape from shape node.
    auto prim_tuple_get_item = std::make_shared<Primitive>(TUPLE_GETITEM_OP);
    AnfNodePtrList inputs{NewValueNode(prim_tuple_get_item), shape_cnode, NewValueNode(MakeValue(SizeToLong(i)))};
    auto tuple_get_item_cnode = func_graph->NewCNode(inputs);
    tuple_get_item_cnode->set_fullname_with_scope("tuple_getitem_replace_reshape");
    prim_tuple_get_item->set_instance_name(instance_name);
    make_tuple_inputs.emplace_back(tuple_get_item_cnode);
    MS_LOG(INFO) << "Create TupleGetItem for " << i;
  }
  auto make_tuple = CreateMakeTuple(make_tuple_inputs, func_graph, instance_name);
  make_tuple->set_in_forward_flag(true);
  std::string fullname = shape_cnode->fullname_with_scope() + "_replace";
  make_tuple->set_fullname_with_scope(fullname);
  manager->SetEdge(reshape_cnode, INDEX_TWO, make_tuple);
  MS_LOG(INFO) << shape_cnode->fullname_with_scope() << "->" << make_tuple->fullname_with_scope() << "->"
               << reshape_cnode->fullname_with_scope();
  MS_LOG(INFO) << "reshape shape is : " << shape;
  MS_LOG(INFO) << "reshape tensor_map is : " << tensor_map.array();
  MS_LOG(INFO) << "reshape dev_arr is : " << dev_arr.array();
  for (size_t i = 0; i < tensor_map.array().size(); ++i) {
    if (tensor_map.GetDimByIdx(i) == -1) {
      continue;
    }
    if (make_tuple_inputs[i]->isa<ValueNode>()) {
      continue;
    }
    int64_t scalar = dev_arr.GetDimByReverseIdx(tensor_map.GetDimByIdx(i));
    Operator scalar_mul_op = CreateScalarMulOp(scalar);
    (void)InsertNode(scalar_mul_op,             // to be inserted op
                     make_tuple,                // current node
                     i + 1,                     // make_tuple[input_index] = scalar_mul_op
                     make_tuple->input(i + 1),  // insert scalar_mul_op between previous and current
                     func_graph,                // current func_graph
                     "update_partial_shape", "", nullptr);
  }
  if (tensor_redistribution != nullptr && tensor_redistribution->original_reshape_shape() != nullptr) {
    tensor_redistribution->set_original_reshape_shape(make_tuple);
    MS_LOG(INFO) << "Change original_reshape_shape";
  }
  return Status::SUCCESS;
}

bool SkipSupplyForReshape(const CNodePtr &cnode) {
  if (!IsReshapeOp(cnode)) {
    return false;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(SKIP_REDISTRIBUTION)) {
    bool skip_redistribution = GetValue<bool>(prim->GetAttr(SKIP_REDISTRIBUTION));
    return skip_redistribution;
  }
  return false;
}

Status UpdateShapeNode(const CNodePtr &cnode, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Step1. Get shape input tensor layout. cnode is Shape op.
  auto input_of_shape = cnode->input(1);
  auto input_cnode = input_of_shape->cast<CNodePtr>();
  if (input_cnode == nullptr) {
    return Status::SUCCESS;
  }
  if (SkipSupplyForReshape(input_cnode)) {
    MS_LOG(INFO) << "Skip " << cnode->fullname_with_scope() << ", because its input is reshape.";
    return Status::SUCCESS;
  }
  if (IsValueNode<FuncGraph>(input_cnode->input(0))) {
    // It means it's a sub-graph call node.
    MS_LOG(WARNING) << "If the input of shape is subgraph, and it's outputs sharding strategy "
                       "is not all 1, it could be problem.";
    return Status::SUCCESS;
  }
  TensorInfo tensor_info;
  if (GetDistributeOperatorFromCNode(input_cnode, &tensor_info) != Status::SUCCESS) {
    return Status::SUCCESS;
  }
  Map tensor_map = tensor_info.tensor_layout().tensor_map();
  Arrangement dev_arr = tensor_info.tensor_layout().device_arrangement();

  // Step2. Get shape node users.
  auto node_users_map = func_graph->manager()->node_users();
  auto shape_node_users = node_users_map[cnode];
  for (const auto &node_user : shape_node_users) {
    MS_EXCEPTION_IF_NULL(node_user.first);
    auto shape_user = node_user.first->cast<CNodePtr>();
    if (IsReshapeOp(shape_user)) {
      std::vector<Shape> input_shapes = GetNodeShape(input_of_shape);
      if (input_shapes.size() != 1) {
        MS_LOG_WITH_NODE(EXCEPTION, shape_user) << "Shape's input size is illegal.";
      }
      if (UpdateReshapeShapeValue(shape_user, cnode, input_shapes[0], tensor_info, func_graph) != Status::SUCCESS) {
        MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Update reshape shape value failed.";
      }
      continue;
    }
    if (shape_user == nullptr || IsTargetOp(shape_user, ZEROS)) {
      MS_LOG(WARNING) << "won't supply shape for " << shape_user->fullname_with_scope();
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(IsTupleGetItem(shape_user),
                               "Only support TupleGetItem here, but got " + GetPrimName(shape_user));
    if (IsTupleGetItem(shape_user) &&
        UpdateTupleGetItemShapeValue(shape_user, tensor_info, func_graph) != Status::SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, shape_user) << "Update tuple get item shape value failed.";
    }
  }
  return Status::SUCCESS;
}

Status MergeEntireShapeForDynamic(const FuncGraphPtr &root) {
  MS_LOG(INFO) << "Into MergeEntireShapeForDynamic";
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
        if (IsShapeOp(cnode)) {
          UpdateShapeNode(cnode, *fg);
          continue;
        }
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
        if (IsShapeOp(cnode)) {
          UpdateShapeNode(cnode, *func_graph);
          continue;
        }
      }
    }
  }
  return Status::SUCCESS;
}

CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node_with_tuple_output,
                                size_t item_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_with_tuple_output);

  auto item_index_value_node = NewValueNode(MakeValue(UlongToLong(item_index)));
  MS_EXCEPTION_IF_NULL(item_index_value_node);

  std::vector<AnfNodePtr> tuple_get_item_inputs = {NewValueNode(prim::kPrimTupleGetItem), node_with_tuple_output,
                                                   item_index_value_node};
  CNodePtr tuple_get_item_node = func_graph->NewCNode(tuple_get_item_inputs);
  MS_EXCEPTION_IF_NULL(tuple_get_item_node);
  const auto &tuple_abstract = node_with_tuple_output->abstract();
  if (!tuple_abstract) {
    return tuple_get_item_node;
  }
  if (!tuple_abstract->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Only create TupleGetItem for tuple output.";
  }
  tuple_get_item_node->set_abstract(tuple_abstract->cast<abstract::AbstractTuplePtr>()->elements()[item_index]);
  return tuple_get_item_node;
}

CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &tuple_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  (void)new_make_tuple_inputs.insert(new_make_tuple_inputs.cend(), tuple_inputs.cbegin(), tuple_inputs.cend());
  auto make_tuple_node = func_graph->NewCNode(new_make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  // MakeTuple's abstract must consist of all inputs' abstract in case unexpected graph compiling error.
  AbstractBasePtrList abstract_list;
  (void)std::for_each(tuple_inputs.cbegin(), tuple_inputs.cend(),
                      [&](const auto &input) { (void)abstract_list.emplace_back(input->abstract()); });
  if (std::find_if(abstract_list.begin(), abstract_list.end(), [](auto abs) { return !abs; }) != abstract_list.end()) {
    return make_tuple_node;
  }
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple_node;
}

}  // namespace mindspore::parallel
