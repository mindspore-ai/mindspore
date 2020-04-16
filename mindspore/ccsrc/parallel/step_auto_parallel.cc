/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "parallel/step_auto_parallel.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "ir/meta_tensor.h"
#include "optimizer/opt.h"
#include "optimizer/optimizer.h"
#include "parallel/auto_parallel/dp_algo_costmodel.h"
#include "parallel/auto_parallel/edge_costmodel.h"
#include "parallel/auto_parallel/graph_costmodel.h"
#include "parallel/auto_parallel/rec_core/rec_generate_strategy.h"
#include "parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "parallel/auto_parallel/rec_core/rec_partition.h"
#include "parallel/context.h"
#include "parallel/ops_info/tmp_identity_info.h"
#include "parallel/step_parallel.h"
#include "pipeline/parse/python_adapter.h"
#include "pipeline/pipeline.h"

namespace mindspore {
namespace parallel {
// splittable_op_ will continuously be updated
std::vector<std::string> splittable_op_ = {MATMUL,
                                           GELU,
                                           TANH,
                                           SOFTMAX,
                                           LOG_SOFTMAX,
                                           ACTIVATION,
                                           PRELU,
                                           FLOORDIV,
                                           L2_NORMALIZE,
                                           TRANSPOSE,
                                           RESHAPE,
                                           TENSOR_ADD,
                                           SUB,
                                           MUL,
                                           DIV,
                                           GREATER,
                                           MAXPOOL,
                                           MAXPOOLV2,
                                           VIRTUAL_DATA_SET,
                                           SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
                                           RELU,
                                           ONEHOT,
                                           DROPOUT_DO_MASK,
                                           REDUCE_MAX,
                                           REDUCE_MIN,
                                           ARGMAXWITHVALUE,
                                           ARGMINWITHVALUE,
                                           REDUCE_SUM,
                                           CONV2D,
                                           FUSE_BATCH_NORM,
                                           POOLING,
                                           SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
                                           SIGMOID_CROSS_ENTROPY_WITH_LOGITS,
                                           MAX_POOL_WITH_ARGMAX,
                                           SIMPLE_MEAN,
                                           FLATTEN,
                                           BATCH_NORM,
                                           LAYER_NORM,
                                           BIAS_ADD,
                                           ASSIGN_SUB,
                                           COS,
                                           ACOS,
                                           EXP,
                                           LOG,
                                           REDUCE_MEAN,
                                           REAL_DIV,
                                           SIGMOID,
                                           POW,
                                           MAXIMUM,
                                           MINIMUM,
                                           EQUAL,
                                           NOT_EQUAL,
                                           LOGICALNOT,
                                           GATHERV2,
                                           STRIDEDSLICE,
                                           SQRT,
                                           GET_NEXT,
                                           CAST,
                                           Neg,
                                           BATCH_MATMUL,
                                           EXPAND_DIMS,
                                           SQUEEZE};

std::vector<std::string> elementwise_op_ = {ACTIVATION, GELU, TANH, SOFTMAX, LOG_SOFTMAX, RELU, SQRT,
                                            CAST,       POW,  EXP,  LOG,     COS,         ACOS, LOGICALNOT};

bool StepAutoParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if ((parallel_mode != AUTO_PARALLEL) || root->flags()[AUTO_PARALLEL_RUN_ONCE_ONLY]) {
    return changes;
  }
  // check whether strategy_search_mode is valid
  std::string strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  if ((strategy_search_mode != DYNAMIC_PROGRAMMING) && (strategy_search_mode != RECURSIVE_PROGRAMMING)) {
    // Setting searching mode: dynanic programming as default.
    strategy_search_mode = DYNAMIC_PROGRAMMING;
    MS_LOG(INFO) << "Non-idicated strategy searching mode, using DP searching mode as default";
  }

  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);

  if (MsContext::GetInstance()->save_graphs_flag()) {
    draw::Draw(STEP_AUTO_PARALLEL_BEGIN, root);
  }
  MS_LOG(INFO) << "Now entering step auto parallel";
  TOTAL_OPS = 0;
  AnfNodePtr ret = root->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }

  // mark the forward cnodes, parallel only care these nodes
  MarkForwardCNode(root);

  if (FindCommunicationOp(all_nodes)) {
    MS_LOG(EXCEPTION) << "The graph contain communication op";
  }

  // search parallelization strategy
  if (strategy_search_mode == DYNAMIC_PROGRAMMING) {
    if (ParallelStrategySearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using DP searching mode";
    }
  } else if (strategy_search_mode == RECURSIVE_PROGRAMMING) {
    if (ParallelStrategyRecSearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using RP searching mode";
    }
  } else {
    MS_LOG(EXCEPTION) << "Auto-parallel strategy searching mode unexpected";
  }

  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Now leaving step auto parallel, used time: " << time << " us";

  root->flags()[AUTO_PARALLEL_RUN_ONCE_ONLY] = true;
  return changes;
}

// Given the node, return whether each input is a parameter or a output of a operator.
// The returned boolean vector should be the same order of the inputs, thus its implementation
// is closely consistent with ExtractShape() in step_parallel.cc
std::vector<bool> ExtractInputParameterByNode(const CNodePtr &node) {
  std::vector<bool> is_parameter;
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  for (size_t i = 1; i < node_inputs.size(); ++i) {
    auto input = node_inputs[i];

    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      if (input_parameter->has_default()) {
        bool require_grad =
          py::cast<bool>(parse::python_adapter::GetPyObjAttr(input_parameter->default_param(), "requires_grad"));
        is_parameter.push_back(require_grad);
      } else {
        is_parameter.push_back(false);
      }
    } else if (input->isa<CNode>() || IsValueNode<tensor::Tensor>(input) || IsValueNode<RefKey>(input)) {
      is_parameter.push_back(false);
    }
  }
  return is_parameter;
}

// Given the type, return the number of bytes to represent this type
size_t GetLengthOfDataType(const TypePtr &type) {
  switch (type->type_id()) {
    case kNumberTypeBool:
      return sizeof(bool);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeInt64:
      return sizeof(int64_t);
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeUInt16:
      return sizeof(uint16_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    case kNumberTypeUInt64:
      return sizeof(uint64_t);
    case kNumberTypeFloat16:
      return sizeof(float) / 2;
    case kNumberTypeFloat32:
      return sizeof(float);
    case kNumberTypeFloat64:
      return sizeof(double);
    case kNumberTypeInt:
      return sizeof(int);
    case kNumberTypeUInt:
      return sizeof(unsigned int);
    case kNumberTypeFloat:
      return sizeof(float);
    default:
      MS_LOG(EXCEPTION) << "Unexpected type " << type->type_name();
  }
}

size_t GetInputsTypeLen(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (!input->isa<CNode>() && !input->isa<Parameter>() && !IsValueNode<tensor::Tensor>(input)) {
    MS_LOG(EXCEPTION) << "The input node is not a cnode or parameter or tensor";
  }

  size_t input_type_len = 0;
  auto type = input->Type();
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<mindspore::TensorType>()) {
    auto input_element_type = type->cast<mindspore::TensorTypePtr>()->element();
    input_type_len = GetLengthOfDataType(input_element_type);
  } else {
    MS_LOG(EXCEPTION) << "Unknown type: " << type->type_name();
  }
  return input_type_len;
}

std::vector<size_t> ExtractInputTypeLengthByNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<size_t> inputs_type_len;
  std::vector<AnfNodePtr> node_inputs{node->inputs()};

  // extract input element length
  for (auto &input : node_inputs) {
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      inputs_type_len.push_back(GetInputsTypeLen(parameters[0]));
    } else if (input->isa<CNode>() || input->isa<Parameter>() || IsValueNode<tensor::Tensor>(input)) {
      // extract input shape from parameter and apply node
      inputs_type_len.push_back(GetInputsTypeLen(input));
    }
  }
  return inputs_type_len;
}

std::vector<TypePtr> ExtractOutputTypeByNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypePtr> outputs_type;
  // extract output element type
  auto primary_output_type = node->Type();
  MS_EXCEPTION_IF_NULL(primary_output_type);
  if (primary_output_type->isa<mindspore::Tuple>()) {
    // in this case, the output is a tuple
    auto tuple_output_type = primary_output_type->cast<mindspore::TuplePtr>();
    auto elements = tuple_output_type->elements();
    for (auto &ele : elements) {
      if (ele->isa<mindspore::TensorType>()) {
        auto ele_element_type = ele->cast<mindspore::TensorTypePtr>()->element();
        outputs_type.push_back(ele_element_type);
      } else {
        MS_LOG(EXCEPTION) << "Unknown type: " << primary_output_type->type_name();
      }
    }
  } else {
    // in this case, the output is a single tensor
    if (primary_output_type->isa<mindspore::TensorType>()) {
      auto element_type = primary_output_type->cast<mindspore::TensorTypePtr>()->element();
      outputs_type.push_back(element_type);
    } else {
      MS_LOG(EXCEPTION) << "Unknown type: " << primary_output_type->type_name();
    }
  }
  return outputs_type;
}

bool IsElementWiseOperator(const std::string &op_name) {
  auto iter = std::find(elementwise_op_.begin(), elementwise_op_.end(), op_name);
  return (iter != elementwise_op_.end());
}

bool IsSplittableOperator(const std::string &op_name) {
  std::vector<std::string>::iterator iter;
  iter = std::find(splittable_op_.begin(), splittable_op_.end(), op_name);
  return (iter != splittable_op_.end());
}

bool IsAutoParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    return false;
  }
  bool bool_result = IsParallelCareNode(cnode) && !IsSplittableOperator(prim->name());
  if (bool_result) {
    MS_LOG(EXCEPTION) << "Should implementing OperatorInfo for: " << prim->name();
  } else if (prim->name() == CAST) {
    return true;
  }
  return IsParallelCareNode(cnode) && IsSplittableOperator(prim->name());
}

OperatorInfoPtr CreateTheOperatorInfo(const PrimitivePtr &prim, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  auto attrs = prim->attrs();
  std::vector<Shapes> shape_list = ExtractShape(cnode);
  if (shape_list.empty()) {
    MS_LOG(EXCEPTION) << "Failure: node " << cnode->UniqueId() << " failed to extract shape";
  }
  // Create an OperatorInfo instance
  OperatorInfoPtr operator_info = NewOperatorInstance(prim, attrs, shape_list);
  MS_EXCEPTION_IF_NULL(operator_info);
  // Set the parameter information for this OperatorInfo (whether the inputs are parameters or not)
  std::vector<bool> parameter_info = ExtractInputParameterByNode(cnode);
  if (operator_info->set_is_parameter(parameter_info) != SUCCESS) {
    MS_LOG(ERROR) << "Initializing parameter information failed for operator: " << operator_info->name();
    return nullptr;
  }
  // Set the data type for inputs and outputs of this OperatorInfo
  auto inputs_type_length = ExtractInputTypeLengthByNode(cnode);
  auto outputs_type = ExtractOutputTypeByNode(cnode);
  std::vector<size_t> outputs_type_length;
  outputs_type_length.reserve(outputs_type.size());
  std::transform(outputs_type.begin(), outputs_type.end(), std::back_inserter(outputs_type_length),
                 GetLengthOfDataType);
  if (operator_info->SetInputAndOutputTypeLength(inputs_type_length, outputs_type_length) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the lengths of inputs and outputs failed for operator: " << operator_info->name();
    return nullptr;
  }
  if (operator_info->set_outputs_type(outputs_type) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the types of outputs failed for operator: " << operator_info->name();
    return nullptr;
  }
  // When the 'inputs' contains numerical values for some operators, these values should be extracted from
  // ANF graph
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>()) {
      input_value.push_back(GetValueNode(inputs[index]));
    } else {
      input_value.emplace_back(nullptr);
    }
  }
  operator_info->set_input_value(input_value);
  operator_info->set_outputs_dtype(cnode->Type());
  operator_info->set_cnode(cnode);
  // If no strategy has been configured for this operator, then candidate strategies are generated for
  // auto-strategy searching; if this primitive is CAST, we ignore the user-specified strategy
  if (!StrategyFound(attrs) || prim->name() == CAST) {
    // Compute split_flag_list_, indicating which input has batch dimension. This is ONLY used for preparation for
    // BatchParallelInfo operator
    operator_info->ComputeBatchSplitFlagList();
    if (operator_info->GenerateStrategies(0) != SUCCESS) {
      MS_LOG(ERROR) << "Strategy search for Operator " << operator_info->name() << " failed.";
      return nullptr;
    }
  } else {
    // In this case, the configured strategy should be extracted to help setting cost
    StrategyPtr strategyPtr = parallel::ExtractStrategy(attrs);
    if (strategyPtr != nullptr) {
      if (prim->name() == RESHAPE) {
        MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
      }
      // Set cost for this configured strategy
      if (operator_info->SetCostUnderStrategy(strategyPtr) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Failure: operator " << prim->name() << " SetCostUnderStrategy failed";
      } else if (FULLY_USE_DEVICES) {
        // If configured to fully use devices, then checking for the user-specified strategy
        int32_t used_devices = operator_info->used_devices();
        MS_EXCEPTION_IF_NULL(g_device_manager);
        auto total_device_num = g_device_manager->GetDeviceListByStageId(0).size();
        // 'used_devices == 1' means that ALL-1 strategy, which is valid in auto-parallel
        if (used_devices == 1) {
          return operator_info;
        }
        // 'used_devices == -1' means that 'used_devices_' is not set
        if ((used_devices == -1) || IntToSize(used_devices) != total_device_num) {
          MS_LOG(EXCEPTION) << "In configuration 'FULLY_USE_DEVICES' = True, "
                            << "but the specified strategy uses device: " << used_devices
                            << ", total devices: " << total_device_num;
        }
      }
    }
  }
  return operator_info;
}

Status ConstructCostGraphNodes(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &) {
  MS_LOG(INFO) << "Constructing nodes for cost graph begins.";
  entire_costgraph = std::make_shared<CostGraph>();
  entire_costgraph->SetDeviceMemoryAndCostParameter();
  bool new_operator = true, first_operator = true;
  std::string first_operator_cnode;
  size_t current_op_index = 0;

  // Step 1
  for (auto &node : all_nodes) {
    // NOTE: we only care about splittable Primitive operators
    auto cnode = node->cast<CNodePtr>();
    bool bool_result = (cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)));
    if (bool_result) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);

    // When visiting the second subgraph, use the corresponding operatorInfo which already created
    bool modify_new_operator = (new_operator) && (!first_operator) && (cnode->UniqueId() == first_operator_cnode);
    if (modify_new_operator) {
      new_operator = false;
    }
    if (new_operator) {
      auto operator_info = CreateTheOperatorInfo(prim, cnode);
      if (operator_info == nullptr) {
        return FAILED;
      }
      // Needed by rec_parser
      operator_info->set_type(prim->name());
      std::vector<std::string> inputs_tensor_name = ExtractInputsTensorName(cnode);
      operator_info->set_cnode_name(cnode->ToString());

      entire_costgraph->AddOperator(operator_info);
      (void)cnode->set_operator_info(operator_info);
      if (first_operator) {
        first_operator_cnode = cnode->UniqueId();
        first_operator = false;
      }
      // Needed by rec_parser
      entire_costgraph->add_inputs_tensor_name(inputs_tensor_name);
    } else {
      auto current_op_ptr = entire_costgraph->FindOperatorByIndex(current_op_index);
      if (current_op_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Find " << prim->name() << " from CostGraph failed.";
      } else {
        bool is_find_wrong = (current_op_ptr->name().find(VIRTUAL_DATA_SET_INFO) == std::string::npos) &&
                             (current_op_ptr->name().find(BATCH_PARALLEL) == std::string::npos) &&
                             (current_op_ptr->name().find(prim->name()) == std::string::npos);
        if (is_find_wrong) {
          MS_LOG(EXCEPTION) << "The OperatorInfo: " << current_op_ptr->name()
                            << " does not match the Prim: " << prim->name();
        }
        (void)cnode->set_operator_info(current_op_ptr);
        current_op_index++;
      }
    }
  }
  if ((!new_operator) && (current_op_index != entire_costgraph->GetOperators().size())) {
    MS_LOG(EXCEPTION) << "The second subgraph's operator number: " << current_op_index
                      << " does not match the first ones: " << entire_costgraph->GetOperators().size();
  }

  MS_LOG(INFO) << "Constructing nodes for cost graph ends.";
  return SUCCESS;
}

void ConstructCostGraphEdges(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 2
  MS_LOG(INFO) << "Constructing edges for cost graph begins.";
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    bool bool_result_cnode = (cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0));
    if (bool_result_cnode) {
      continue;
    }
    auto &inputs = cnode->inputs();
    ValueNodePtr prim_anf_node = inputs[0]->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    size_t edge_count = 0;

    for (size_t i = 1; i < inputs.size(); ++i) {
      auto prev_cnode = inputs[i]->cast<CNodePtr>();
      bool bool_result_prev_cnode = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
      if (bool_result_prev_cnode) {
        continue;
      }
      ValueNodePtr prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
      PrimitivePtr prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
      size_t output_index = 0;

      bool bool_result =
        (IsAutoParallelCareNode(prev_cnode)) || (prev_prim->name() == TUPLE_GETITEM) || (prev_prim->name() == DEPEND);
      while (bool_result) {
        if (IsAutoParallelCareNode(prev_cnode)) {
          std::string edge_name =
            prev_cnode->operator_info()->name() + OPERATOR_TO_OPERATOR_CONNECTOR + cnode->operator_info()->name();
          // If the edge between these two operators already has been added, then the edge will not be added again.
          if (entire_costgraph->IsEdgeInCostGraph(edge_name, output_index, i - 1)) {
            break;
          }
          EdgePtr edge_ptr;
          MS_LOG(INFO) << "Creating edge: " << edge_name;

          bool follow_strategy = ELEMENTWISE_OP_STRA_FOLLOW && IsElementWiseOperator(prev_prim->name());
          if (follow_strategy) {
            // Redistribution in not allowed on the edge.
            // Elementwise operators have the same strategy as their previous operators.
            edge_ptr = std::make_shared<Edge>(edge_name, prev_cnode->operator_info(), cnode->operator_info(),
                                              output_index, i - 1, false, true);
          } else {
            edge_ptr = std::make_shared<Edge>(edge_name, prev_cnode->operator_info(), cnode->operator_info(),
                                              output_index, i - 1, false);
          }

          // Init costs for this edge
          if (edge_ptr->InitEdgeCost() != SUCCESS) {
            MS_LOG(EXCEPTION) << "Edge cost initialization failed";
          }
          cnode->operator_info()->AddPrevEdge(edge_ptr);
          prev_cnode->operator_info()->AddSuccEdge(edge_ptr);
          entire_costgraph->AddEdge(prev_cnode->operator_info(), cnode->operator_info(), edge_ptr);
          MS_LOG(INFO) << "Successfully adding the edge between " << prev_cnode->operator_info()->name() << " and "
                       << cnode->operator_info()->name();
          edge_count++;

          break;
        } else if (prev_prim->name() == TUPLE_GETITEM) {
          // In this case, 'prev_anf_node' is 'tuple_getitem', the actual precursor node is node before
          // this 'tuple_getitem'
          MS_LOG(INFO) << "Jumping the 'tuple_getitem' operator.";
          output_index = IntToSize(GetValue<int>(GetValueNode(prev_cnode->input(2))));
          prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
          bool bool_result_tuple = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
          if (bool_result_tuple) {
            break;
          }
          prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
          prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
          if (!IsAutoParallelCareNode(prev_cnode)) {
            MS_LOG(EXCEPTION) << "Did not create OperatorInfo for : " << prev_prim->name();
          }
          MS_LOG(INFO) << "Jumped the 'tuple_getitem' operator, "
                       << "and creating an edge between the Operator before "
                       << "'tuple_getitem' and the Operator after 'tuple_getitem'.";
        } else if (prev_prim->name() == DEPEND) {
          // In this case, 'prev_anf_node' is 'depend', the actual precursor node is node before
          // this 'depend'
          MS_LOG(INFO) << "Jumping the 'depend' operator.";
          prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
          bool bool_result_depend = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
          if (bool_result_depend) {
            break;
          }
          prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
          prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
          MS_LOG(INFO) << "Jumped the 'depend' operator, "
                       << "and creating an edge between the Operator before "
                       << "'depend' and the Operator after 'depend'.";
        }
        bool_result =
          (IsAutoParallelCareNode(prev_cnode)) || (prev_prim->name() == TUPLE_GETITEM) || (prev_prim->name() == DEPEND);
      }
    }
    MS_LOG(INFO) << "Successfully created " << edge_count << " edges for: " << cnode->operator_info()->name();
  }

  MS_LOG(INFO) << "Constructing edges for cost graph ends.";
}

std::pair<AnfNodePtr, std::vector<AnfNodePtr>> CNodeWithRefKeys(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> refkeys;
  if (cnode->isa<CNode>()) {
    auto cnode_ptr = cnode->cast<CNodePtr>();
    auto inputs = cnode_ptr->inputs();
    for (auto &one_input : inputs) {
      if (IsValueNode<RefKey>(one_input)) {
        refkeys.push_back(one_input);
      }
    }
    if (refkeys.size() >= 1) {
      return std::make_pair(cnode, refkeys);
    }
  }
  return {nullptr, refkeys};
}

void AugmentCostGraph(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 3
  for (auto &node : all_nodes) {
    auto cnode_with_refkeys = CNodeWithRefKeys(node);
    if ((!node->isa<Parameter>()) && (cnode_with_refkeys.first == nullptr)) {
      continue;
    }
    std::string parameter_name;
    AnfNodePtr target_parameter = nullptr;
    AnfNodeIndexSet target_set;

    if (cnode_with_refkeys.first != nullptr) {
      // Dealing with the RefKey case
      auto refkeys = cnode_with_refkeys.second;
      auto cnode = cnode_with_refkeys.first;

      auto cnode_ptr = cnode->cast<CNodePtr>();
      if (cnode_ptr == nullptr || !IsValueNode<Primitive>(cnode_ptr->input(0))) {
        continue;
      }
      if (!IsAutoParallelCareNode(cnode_ptr)) {
        continue;
      }

      if (refkeys.size() > 1) {
        MS_LOG(EXCEPTION) << "CNode: " << cnode->fullname_with_scope() << " 's inputs have more than 1 RefKeys.";
      }
      MS_EXCEPTION_IF_NULL(cnode->func_graph());
      auto cnode_func_graph = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(cnode->func_graph()->manager());

      // Find the RefKey being used
      auto candidate_set_by_refkey = cnode_func_graph->manager()->node_users()[refkeys[0]];
      for (auto &candidate : candidate_set_by_refkey) {
        auto candidate_node = candidate.first;
        auto c = candidate_node->cast<CNodePtr>();
        if (c == nullptr || !IsValueNode<Primitive>(c->input(0))) {
          continue;
        }
        if (!IsAutoParallelCareNode(c)) {
          continue;
        }
        target_set.add(candidate);
      }

      // Find the corresponding Parameter being used
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(refkeys[0], cnode_func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      parameter_name = parameters[0]->cast<ParameterPtr>()->name();
      target_parameter = parameters[0];
      auto candidate_set_by_para = cnode_func_graph->manager()->node_users()[parameters[0]];
      for (auto &candidate : candidate_set_by_para) {
        auto candidate_node = candidate.first;
        auto c = candidate_node->cast<CNodePtr>();
        if (c == nullptr || !IsValueNode<Primitive>(c->input(0))) {
          continue;
        }
        if (!IsAutoParallelCareNode(c)) {
          continue;
        }
        (void)target_set.insert(candidate);
      }
    } else if (node->isa<Parameter>()) {
      // Dealing with the Parameter case
      MS_EXCEPTION_IF_NULL(node->func_graph());
      MS_EXCEPTION_IF_NULL(node->func_graph()->manager());
      auto candidate_set = node->func_graph()->manager()->node_users()[node];
      for (auto &candidate : candidate_set) {
        auto candidate_node = candidate.first;
        auto c = candidate_node->cast<CNodePtr>();
        if (c == nullptr || !IsValueNode<Primitive>(c->input(0))) {
          continue;
        }
        if (!IsAutoParallelCareNode(c)) {
          continue;
        }
        (void)target_set.insert(candidate);
      }
      // In this case, node is a Parameter
      parameter_name = node->cast<ParameterPtr>()->name();
      target_parameter = node;
    }
    if (target_set.size() <= 1) {
      continue;
    }

    // Rule out the case when a Parameter being used by a Operator, but the Operator appears in multiple CNODEs
    std::set<std::string> target_without_duplicate;
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      auto input_index = target.second;
      (void)target_without_duplicate.insert(std::to_string(input_index) + target_cnode->operator_info()->name());
    }
    if (target_without_duplicate.size() <= 1) {
      continue;
    }

    // Here, it is sure that this Parameter (RefKey) is being used by multiple Operators.
    OperatorInfoPtr tmp_identity_ptr;
    bool new_identity = false;
    std::string tmp_identity_name;
    auto returned_identity = entire_costgraph->FindTmpIdentityByParameterName(parameter_name);
    if (returned_identity != nullptr) {
      // In this case, the TmpIdentityInfo instance has already been created
      new_identity = false;
      tmp_identity_ptr = returned_identity;
      tmp_identity_name = tmp_identity_ptr->name();
    } else {
      // In the case, the TmpIdentityInfo instance has NOT been created. Thus, a new one is created.
      new_identity = true;
      // 1) extract input shape from this Parameter
      MS_EXCEPTION_IF_NULL(target_parameter);
      AbstractBasePtr abstract = target_parameter->abstract();
      if (abstract == nullptr) {
        MS_LOG(EXCEPTION) << "Failure: abstract is nullptr";
      }
      auto input_shape = dyn_cast<abstract::Shape>(abstract->GetShapeTrack());
      if (input_shape == nullptr) {
        MS_LOG(EXCEPTION) << "Failure: input_shape is nullptr";
      }
      std::vector<int> shape_int = input_shape->shape();
      Shape shape;
      (void)std::transform(shape_int.begin(), shape_int.end(), std::back_inserter(shape),
                           [](int sub_shape) { return static_cast<int32_t>(sub_shape); });
      Shapes inputs_shape = {shape};
      Shapes outputs_shape = {shape};
      // 2) init the attr
      std::unordered_map<std::string, ValuePtr> attr = {};

      // Create the TmpIdentity instance
      tmp_identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
      tmp_identity_ptr->set_name(tmp_identity_ptr->name() + std::to_string(TOTAL_OPS));
      TOTAL_OPS++;
      tmp_identity_ptr->set_refkey_parameter_name(parameter_name);
      // Set the parameter and type lengths for inputs and outputs
      std::vector<bool> is_parameter;
      auto casted_target_parameter = target_parameter->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(casted_target_parameter);
      if (casted_target_parameter->has_default()) {
        bool require_grad = py::cast<bool>(
          parse::python_adapter::GetPyObjAttr(casted_target_parameter->default_param(), "requires_grad"));
        is_parameter.push_back(require_grad);
      } else {
        is_parameter.push_back(false);
      }
      if (tmp_identity_ptr->set_is_parameter(is_parameter) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Setting parameter for TmpIdentityInfo failed";
      }
      auto node_type = target_parameter->Type();
      if (node_type->isa<mindspore::TensorType>()) {
        auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
        std::vector<size_t> type_length = {GetLengthOfDataType(input_element_type)};
        if (tmp_identity_ptr->SetInputAndOutputTypeLength(type_length, type_length) != SUCCESS) {
          MS_LOG(EXCEPTION) << "Setting input and output type length for TmpIdentityInfo failed";
        }
      } else {
        MS_LOG(EXCEPTION) << "Unknown type: " << node_type->type_name();
      }

      // Generate strategies for this TmpIdentityInfo instance;
      if (tmp_identity_ptr->GenerateStrategies(0) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Strategy search for Operator failed : " << tmp_identity_ptr->name();
      }
    }
    // A flag recording whether new edges have been created or not
    bool add_identity_edge = false;

    // Create edges between this TmpIdentityInfo instance and subsequent Operator instances
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      auto prim = GetValueNode<PrimitivePtr>(target_cnode->input(0));
      auto input_index = target.second;

      std::string edge_name =
        std::string(IDENTITY_INFO) + OPERATOR_TO_OPERATOR_CONNECTOR + target_cnode->operator_info()->name();
      // If the edge between these two operators already has been added, then the edge will not be added again.
      if (entire_costgraph->IsEdgeInCostGraph(edge_name, 0, IntToSize(input_index - 1))) {
        continue;
      }
      std::shared_ptr<Edge> edge_ptr = std::make_shared<Edge>(
        edge_name, tmp_identity_ptr, target_cnode->operator_info(), 0, input_index - 1, false, true);

      if (edge_ptr->InitEdgeCost() != SUCCESS) {
        MS_LOG(EXCEPTION) << "Edge cost initialization failed";
      }
      target_cnode->operator_info()->AddPrevEdge(edge_ptr);
      tmp_identity_ptr->AddSuccEdge(edge_ptr);
      entire_costgraph->AddEdge(tmp_identity_ptr, target_cnode->operator_info(), edge_ptr);
      MS_LOG(INFO) << "Successfully adding the edge between " << tmp_identity_ptr->name() << " and "
                   << target_cnode->operator_info()->name();
      add_identity_edge = true;
    }
    if (new_identity && add_identity_edge) {
      // Add the TmpIdentityInfo to CostGraph if BOTH two conditions are satisfied
      entire_costgraph->AddOperator(tmp_identity_ptr);
    }
  }
}

Status ParallelStrategySearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  // There are 4 meta-steps to determine the parallelization strategy for the ANF graph.
  // Step 1: Traverse the ANF graph, and create NODEs for costgraph:
  //      create the OperatorInfo object for each primitive, and enumerate the parallelization strategies
  //      for each OperatorInfo;
  // Step 2: Traverse the ANF graph, and create EDGES for costgraph:
  //      create the Edge object for each pair of OperatorInfo, and enumerate the parallelization strategies
  //      for each edge, based on the strategies of two OperatorInfos;
  // Step 3: Augment the costgraph:
  //      taking care for the case of a single Parameter being used by multiple operators. Create a TmpIdentity
  //      operator for this Parameter, and add an edge for the use of this Parameter by each
  //      subsequent operator;
  // Step 3.1: Calculate memory usage
  // Step 4: Run the Dynamic Programming algorithm:
  //      in this process, cost is calculated based on not only the operators, but also the edges. Here, the edge
  //      cost is caused by the redistribution of a operator's output tensor layout to the next operator's input
  //      tensor layout. Note that there may be several connected components in the costgraph, and the DP algorithm
  //      runs on each of them.
  //
  // OUTPUT: the determined strategy for each operator.

  // Step 1
  if (ConstructCostGraphNodes(all_nodes, root) == SUCCESS) {
    MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are " << entire_costgraph->GetOperators().size()
                 << " operators.";
  } else {
    MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
  }

  // Step 2
  ConstructCostGraphEdges(all_nodes);
  MS_LOG(INFO) << "Constructing edges for cost graph succeeded. There are " << entire_costgraph->GetOperators().size()
               << " operators, and " << entire_costgraph->GetNumPairs() << " edges.",

    // Step 3: Augment the costgraph.
    AugmentCostGraph(all_nodes);
  MS_LOG(INFO) << "After the augmenting procedure, there are " << entire_costgraph->GetOperators().size()
               << " operators, and " << entire_costgraph->GetNumPairs() << " edges.";

  // Step 3.1: Calculate the memory usage
  if (entire_costgraph->ComputeOpsAndEdgesParameterInvolved() == SUCCESS) {
    // Calculate operators' memory usage
    if (entire_costgraph->CalculateOpsMemoryCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Calculating operators' cost for memory cost failed.";
    }
    // Calculate edges' memory usage
    if (entire_costgraph->CalculateEdgesMemoryCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Calculating edges' cost for memory cost failed.";
    }
    // Correct memory usage caused by TmpIdentity
    if (entire_costgraph->CorrectOpsMemoryCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Correcting operators' cost for memory cost failed.";
    }
  } else {
    MS_LOG(EXCEPTION) << "Computing operators' parameter_involved failed.";
  }

  // Step 4: run DP algorithm on the costgraph.
  if (GetStrategy(entire_costgraph) != SUCCESS) {
    MS_LOG(ERROR) << "Strategy search for cost-graph fails";
    return FAILED;
  }
  MS_LOG(INFO) << "Searching strategy succeeded.";

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(EXCEPTION) << "Init selected strategy failed.";
  }

  // print the selected strategy
  for (auto &op : entire_costgraph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    MS_LOG(INFO) << op->name() << " : The strategy is:";
    PrintStrategy(s_strategy);
  }

  return SUCCESS;
}

std::vector<std::vector<std::string>> RecInputTensorNames(const std::map<std::string, std::string>::iterator &it,
                                                          std::vector<std::vector<std::string>> input_tensor_names) {
  for (size_t j = 0; j < input_tensor_names.size(); j++) {
    for (size_t k = 0; k < input_tensor_names[j].size(); k++) {
      if (it->first == input_tensor_names[j][k]) {
        input_tensor_names[j][k] = it->second;
        break;
      }
    }
  }
  return input_tensor_names;
}

Status ParallelStrategyRecSearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  if (ConstructCostGraphNodes(all_nodes, root) == SUCCESS) {
    MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are " << entire_costgraph->GetOperators().size()
                 << " operators.";
  } else {
    MS_LOG(ERROR) << "Constructing nodes for cost graph failed.";
    return FAILED;
  }
  auto ops = entire_costgraph->GetOperators();
  std::vector<std::vector<std::string>> input_tensor_names = entire_costgraph->get_inputs_tensor_name_list();
  auto tuple_getitem_list = entire_costgraph->get_tuple_getitem_list();
  for (auto it = tuple_getitem_list.begin(); it != tuple_getitem_list.end();) {
    input_tensor_names = RecInputTensorNames(it++, input_tensor_names);
  }

  std::shared_ptr<std::vector<size_t>> ops_nodes_list(new std::vector<size_t>);

  std::shared_ptr<Graph> graph = ParseGraph(ops, input_tensor_names, ops_nodes_list);

  size_t num_device = g_device_manager->DeviceNum();
  if (PartitionForAllDevices(num_device, graph) == SUCCESS) {
    MS_LOG(INFO) << "Partition Success With " << num_device << " devices.";
  } else {
    MS_LOG(ERROR) << "PartitionForAllDevices failed.";
    return FAILED;
  }

  bool mask_special_ops = true;
  GenerateStrategy(graph, mask_special_ops, ops);

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(ERROR) << "Init selected strategy failed.";
    return FAILED;
  }

  // print the selected strategy
  for (auto &op : entire_costgraph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    MS_LOG(INFO) << op->name() << " : The strategy is:";
    PrintStrategy(s_strategy);
  }

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
