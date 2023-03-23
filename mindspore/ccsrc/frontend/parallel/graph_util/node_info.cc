/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/node_info.h"

#include <string>
#include <utility>

#include "mindspore/core/ops/core_ops.h"
#include "ir/param_info.h"
#include "ir/meta_tensor.h"
#include "include/common/utils/python_adapter.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
const std::vector<std::string> filter_attrs = {RECOMPUTE, TARGET};
std::string ParameterName(const AnfNodePtr &node_ptr) {
  auto para_ptr = node_ptr->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para_ptr);
  return para_ptr->name();
}

bool ParameterRequireGrad(const AnfNodePtr &node_ptr) {
  auto para_ptr = node_ptr->cast<ParameterPtr>();
  if (para_ptr == nullptr) {
    return false;
  }
  if (!para_ptr->has_default()) {
    return false;
  }
  auto param_value = para_ptr->param_info();
  if (param_value == nullptr) {
    return false;
  }
  return param_value->requires_grad();
}

AnfNodePtr GetRealInput(const AnfNodePtr &input) {
  auto res = input;
  while (IsPrimitiveCNode(res, prim::kPrimLoad) || IsPrimitiveCNode(res, prim::kPrimDepend)) {
    res = res->cast<CNodePtr>()->input(1);
    if (!res->isa<CNode>()) {
      return res;
    }
  }
  return res;
}

// Given the node, return whether each input is a parameter or a output of a operator.
// The returned boolean vector should be the same order of the inputs, thus its implementation
// is closely consistent with ExtractShape() in step_parallel.cc
std::vector<bool> ExtractInputParameterByNode(const CNodePtr &node) {
  std::vector<bool> is_parameter;
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  // input is a ValueList or ValueTuple, then all inputs are not parameter.
  if ((node_inputs.size() == 2) &&
      (IsValueNode<ValueList>(node_inputs[1]) || IsValueNode<ValueTuple>(node_inputs[1]))) {
    std::vector<ValuePtr> inputs_seq;
    if (IsValueNode<ValueList>(node_inputs[1])) {
      inputs_seq = node_inputs[1]->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
    } else {
      inputs_seq = node_inputs[1]->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
    }
    size_t inputs_seq_tensor_size = inputs_seq.size();
    for (const auto &inputs_seq_value : inputs_seq) {
      auto tensor = inputs_seq_value->cast<tensor::TensorPtr>();
      if (tensor == nullptr) {
        MS_LOG(DEBUG) << "The value not is not a tensor.";
        inputs_seq_tensor_size = 0;
        break;
      }
    }
    return std::vector<bool>(inputs_seq_tensor_size, false);
  }
  if ((node_inputs.size() == 2) &&
      (AnfNodeIsPrimitive(node_inputs[1], MAKE_TUPLE) || AnfNodeIsPrimitive(node_inputs[1], MAKE_LIST))) {
    node_inputs = node_inputs[1]->cast<CNodePtr>()->inputs();
  }
  for (size_t i = 1; i < node_inputs.size(); ++i) {
    auto input = GetRealInput(node_inputs[i]);
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      is_parameter.push_back(ParameterRequireGrad(input_parameter));
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
      return sizeof(int64_t);
    case kNumberTypeUInt:
      return sizeof(unsigned);
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

  if ((node_inputs.size() == 2) &&
      (IsValueNode<ValueList>(node_inputs[1]) || IsValueNode<ValueTuple>(node_inputs[1]))) {
    std::vector<ValuePtr> inputs_seq;
    if (IsValueNode<ValueList>(node_inputs[1])) {
      inputs_seq = node_inputs[1]->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
    } else {
      inputs_seq = node_inputs[1]->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
    }
    for (auto &ele : inputs_seq) {
      auto tensor = ele->cast<tensor::TensorPtr>();
      if (tensor == nullptr) {
        inputs_type_len.clear();
        return inputs_type_len;
      }
      inputs_type_len.push_back(GetLengthOfDataType(tensor->Dtype()));
    }
    return inputs_type_len;
  }

  if ((node_inputs.size() == 2) &&
      (AnfNodeIsPrimitive(node_inputs[1], MAKE_TUPLE) || AnfNodeIsPrimitive(node_inputs[1], MAKE_LIST))) {
    node_inputs = node_inputs[1]->cast<CNodePtr>()->inputs();
  }

  // extract input element length
  for (auto &input : node_inputs) {
    if (HasAbstractMonad(input)) {
      continue;
    }
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

std::vector<AnfNodePtr> FindParameterByRefKeyNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> parameters;
  if (!IsValueNode<RefKey>(node)) {
    MS_LOG(ERROR) << "The node is not a ref key";
    return parameters;
  }

  auto ref_key = GetValueNode<StringImmPtr>(node);
  MS_EXCEPTION_IF_NULL(ref_key);
  auto name = ref_key->value();

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto roots = manager->roots();
  if (roots.size() != 1) {
    MS_LOG(ERROR) << "The size of roots ( " << roots.size() << " ) is not 1";
    return parameters;
  }

  FuncGraphPtr root_g = roots.back();
  MS_EXCEPTION_IF_NULL(root_g);
  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param && (name == param->name())) {
      parameters.push_back(param_node);
      MS_LOG(INFO) << "The name of ref key is: " << name;
      return parameters;
    }
  }

  MS_LOG(ERROR) << "The name of ref key is: " << name << ", but have not found the parameter";
  return parameters;
}

bool AnfNodeIsPrimitive(const AnfNodePtr &anf_node, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == prim_name) {
    return true;
  }
  return false;
}

bool FindReshape(const CNodePtr &cnode, mindspore::HashSet<std::string> *op_cache) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) {
    return false;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == RESHAPE) {
    auto operator_info = cnode->user_data<OperatorInfo>();
    std::string op_info_name = operator_info->name();
    if (op_cache->find(op_info_name) != op_cache->end()) {
      return false;
    }
    (void)op_cache->insert(op_info_name);
    return true;
  }
  return false;
}

// Find previous node of Reshape, then obtain its strategy_cost_ vector to get its layout vector.
bool FindReshapePreNodeStraCosts(const AnfNodePtr &node, OperatorInfoPtr *pre_operator_info, int64_t *out_index,
                                 size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding Reshape's previous node, exceeded the max recursive depth: "
                    << MAX_RECURSIVE_DEPTH;
    return false;
  }
  // if previous node is a parameter, handle it in the outsize.
  if (node->isa<Parameter>()) {
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  FindPreNodeCrossFuncGraph(&cnode, out_index);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  auto node_op_info = cnode->user_data<OperatorInfo>();
  if (IsParallelCareNode(cnode) && (node_op_info != nullptr) && !IsPrimitiveCNode(cnode, prim::kPrimReshape)) {
    *pre_operator_info = node_op_info;
    *out_index = 0;
    return true;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == prim::kTupleGetItem) {
    *out_index = GetTupleGetItemIndex(cnode);
    // find tuple_get_item's previous node
    auto pre_node = cnode->input(1);
    if (!pre_node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "tuple get item's second input is not a cnode";
    }
    CNodePtr pre_cnode = pre_node->cast<CNodePtr>();
    FindPreNodeCrossFuncGraph(&pre_cnode, out_index);
    auto pre_op_info = pre_cnode->user_data<OperatorInfo>();
    if (IsParallelCareNode(pre_cnode) && (pre_op_info != nullptr)) {
      *pre_operator_info = pre_op_info;
      return true;
    }
    return false;
  }
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    if (!FindReshapePreNodeStraCosts(cnode->inputs()[index], pre_operator_info, out_index, ++curr_depth)) {
      continue;
    }
    return true;
  }
  MS_LOG(WARNING)
    << "FindReshapePreNodeStraCosts failed, if reshape is not the first primitive, there must be some error";
  return false;
}

// Find next node of Reshape, then obtain its strategy_cost_ vector to get its layout vector.
// if reshape's output connect to several primitive, return the first layout found
void FindReshapeNextNodeStraCosts(const CNodePtr &cnode,
                                  std::vector<std::pair<OperatorInfoPtr, int64_t>> *next_ops_index,
                                  bool *is_next_reshape, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding Reshape's next node, exceeded the max recursive depth: " << MAX_RECURSIVE_DEPTH;
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[cnode];
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr ||
        !(IsValueNode<Primitive>(use_apply->input(0)) || IsValueNode<FuncGraph>(use_apply->input(0)))) {
      continue;
    }
    auto pair = node_pair;
    if (IsValueNode<FuncGraph>(use_apply->input(0))) {
      auto sub_graph = GetValueNode<FuncGraphPtr>(use_apply->input(0));
      auto params = sub_graph->parameters();
      auto sub_manager = sub_graph->manager();
      auto sub_node_set = sub_manager->node_users()[params[node_pair.second - 1]];
      for (auto &sub_node_pair : sub_node_set) {
        use_apply = sub_node_pair.first->cast<CNodePtr>();
        pair = sub_node_pair;
        break;
      }
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimReshape)) {
      *is_next_reshape = true;
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    MS_LOG(INFO) << "FindNextLayout prim " << node_prim->name();
    if (node_prim->name() == DEPEND && pair.second != 1) {
      continue;
    }
    auto op_info = use_apply->user_data<OperatorInfo>();
    if (IsParallelCareNode(use_apply) && (op_info != nullptr)) {
      MS_LOG(INFO) << "FindReshapeNextNodeStraCosts success prim " << node_prim->name();
      *is_next_reshape = false;
      next_ops_index->push_back(std::make_pair(op_info, pair.second - 1));
      continue;
    }
    MS_LOG(DEBUG) << "FindReshapeNextNodeStraCosts failed prim " << node_prim->name() << "  "
                  << IsParallelCareNode(use_apply) << "   " << (op_info != nullptr);

    FindReshapeNextNodeStraCosts(use_apply, next_ops_index, is_next_reshape, ++curr_depth);
  }
}

void SetUserAttrs(const mindspore::HashMap<std::string, ValuePtr> &origin_prim_attrs, const PrimitivePtr &self_prim) {
  MS_EXCEPTION_IF_NULL(self_prim);
  for (auto attr_name : filter_attrs) {
    auto iter = origin_prim_attrs.find(attr_name);
    if (iter != origin_prim_attrs.cend()) {
      self_prim->set_attr(attr_name, iter->second);
      MS_LOG(INFO) << "The new prim " << self_prim << " add attr " << attr_name;
    }
  }
}

// Convert ValueTuple/ValueList to vector
Status TransValueSequeueToVector(const ValuePtr &input_value, std::vector<int64_t> *input) {
  MS_EXCEPTION_IF_NULL(input_value);
  input->clear();
  if (!input_value->isa<ValueSequeue>()) {
    MS_LOG(ERROR) << "Input value must be ValueTuplePtr.";
    return FAILED;
  }
  ValueSequeuePtr value_seq = input_value->cast<ValueSequeuePtr>();
  for (auto &element : value_seq->value()) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t value = element->cast<Int64ImmPtr>()->value();
      input->push_back(value);
    } else {
      MS_LOG(ERROR) << "The value must be int64";
      return FAILED;
    }
  }
  return SUCCESS;
}

// Get the input of cnode, skipping DEPEND/LOAD/UPDATESTATE
const AnfNodePtr RealInputNode(const CNodePtr cnode, size_t index) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() <= index) {
    MS_LOG(EXCEPTION) << "cnode inputs size: " << cnode->size() << " is less equal index: " << index;
  }
  auto input0 = cnode->input(index);
  if (!input0->isa<CNode>()) {
    return input0;
  }
  auto prim = GetCNodePrimitive(input0);
  MS_EXCEPTION_IF_NULL(prim);
  while (prim->name() == LOAD || prim->name() == DEPEND || prim->name() == UPDATESTATE) {
    if (prim->name() == LOAD || prim->name() == DEPEND) {
      input0 = input0->cast<CNodePtr>()->input(1);
    } else {
      input0 = input0->cast<CNodePtr>()->input(2);
    }
    if (!input0->isa<CNode>()) {
      return input0;
    }
    prim = GetCNodePrimitive(input0);
    MS_EXCEPTION_IF_NULL(prim);
  }
  return input0;
}
}  // namespace parallel
}  // namespace mindspore
