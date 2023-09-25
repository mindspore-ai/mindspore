/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/bprop_pass.h"
#include <utility>
#include <memory>
#include <vector>
#include "pipeline/pynative/pynative_utils.h"
#include "ops/sequence_ops.h"
#include "ops/nn_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace pynative {
namespace bprop_pass {
namespace {
constexpr auto kTupleToMakeTuple = "tuple_to_make_tuple";

mindspore::HashMap<std::string, std::vector<std::pair<size_t, ValuePtr>>> node_attr_value_;

void RevertMakeTupleNode(const FuncGraphPtr &tape_graph, const CNodePtr &cnode, ValuePtrList *input_value,
                         AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  MS_EXCEPTION_IF_NULL(input_value);
  if (!cnode->HasAttr(kTupleToMakeTuple)) {
    return;
  }
  AnfNodePtrList new_inputs{cnode->input(kIndex0)};
  const auto &dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
  for (size_t i = 0; i < dyn_input_sizes.size(); ++i) {
    if (dyn_input_sizes[i] >= 0) {
      // Compress input
      AnfNodePtrList cnode_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
      AnfNodePtrList knode_inputs{NewValueNode(prim::kPrimMakeTuple)};
      ValuePtrList value_tuple;
      abstract::AbstractBasePtrList abs_list;
      for (int64_t j = 0; j < dyn_input_sizes[i]; ++j) {
        auto input = cnode->input(i + j + kIndex1);
        (void)cnode_tuple_inputs.emplace_back(input);
        (void)knode_inputs.emplace_back(cnode_inputs->at(i + j + kIndex1));
        (void)value_tuple.emplace_back(input_value->at(i + j));
        (void)abs_list.emplace_back(input->abstract());
      }
      // Update knode inputs to make tuple inputs
      auto cnode_graph = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(cnode_graph);
      auto cnode_tuple = cnode_graph->NewCNode(cnode_tuple_inputs);
      auto abs = std::make_shared<abstract::AbstractTuple>(abs_list);
      cnode_tuple->set_abstract(abs);
      (void)new_inputs.emplace_back(cnode_tuple);

      // Update knode inputs
      auto knode_input = tape_graph->NewCNode(knode_inputs);
      knode_input->set_abstract(abs);
      size_t begin_index = i + kIndex1;
      (void)cnode_inputs->erase(cnode_inputs->begin() + SizeToLong(begin_index),
                                cnode_inputs->begin() + SizeToLong(begin_index) + dyn_input_sizes[i]);
      (void)cnode_inputs->emplace_back(knode_input);

      // Update input value
      (void)input_value->erase(input_value->begin() + SizeToLong(begin_index),
                               input_value->begin() + SizeToLong(begin_index) + dyn_input_sizes[i]);
      (void)input_value->emplace_back(std::make_shared<ValueTuple>(value_tuple));
    } else {
      (void)new_inputs.emplace_back(cnode->input(i + kIndex1));
    }
  }
  cnode->set_inputs(new_inputs);
  cnode->EraseAttr(kTupleToMakeTuple);
}

void TraverseCnode(const CNodePtr &cnode, const std::string &device_target, bool is_dynamic_shape, bool grad_by_value) {
  for (size_t i = 1; i < cnode->size(); ++i) {
    // Avoiding infinite loops
    if (!cnode->HasAttr(kIsKNode) && cnode->input(i)->isa<CNode>()) {
      cnode->set_input(
        i, ConvertConstInputToAttr(cnode->input(i)->cast<CNodePtr>(), device_target, is_dynamic_shape, grad_by_value));
    }
  }
}

void ProcessValueNode(const ValueNodePtr &v_node) {
  MS_EXCEPTION_IF_NULL(v_node);
  const auto &value = v_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->type();
  if (PyNativeAlgo::Common::IsTensor(value, true) || value->isa<Number>() || value->isa<None>() ||
      (type != nullptr && type->isa<String>())) {
    return;
  }
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else if (value->isa<ValueList>()) {
    tensor_ptr = opt::CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value()));
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple, but get type " << value->type_name()
                      << ", value " << value->ToString();
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  v_node->set_value(tensor_ptr);
  v_node->set_abstract(tensor_ptr->ToAbstract());
}
}  // namespace

void ConvertValueNodeValueToTensor(const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(din);
  if (din->isa<CNode>()) {
    const auto &cnode = din->cast<CNodePtr>();
    if (cnode->HasAttr(kIsKNode) || IsPrimitiveCNode(cnode, prim::kPrimBpropCut)) {
      return;
    }
    if (IsPrimitiveCNode(din, prim::kPrimTupleGetItem)) {
      ConvertValueNodeValueToTensor(cnode->input(kIndex1));
      return;
    }
    mindspore::HashSet<size_t> none_inputs;
    for (size_t i = 1; i < cnode->size(); ++i) {
      if (cnode->input(i)->isa<ValueNode>() && cnode->input(i)->cast<ValueNodePtr>()->value()->isa<None>()) {
        (void)none_inputs.insert(i);
        continue;
      }
      ConvertValueNodeValueToTensor(cnode->input(i));
    }
    if (!none_inputs.empty()) {
      AnfNodePtrList new_inputs;
      for (size_t i = kIndex0; i < cnode->size(); ++i) {
        if (none_inputs.count(i) == 0) {
          new_inputs.emplace_back(cnode->input(i));
        }
      }
      cnode->set_inputs(new_inputs);
    }
  } else if (din->isa<ValueNode>()) {
    ProcessValueNode(din->cast<ValueNodePtr>());
  }
}

void ConvertMakeTupleInputToDynamicInput(const AnfNodePtr &node, SeenNum seen,
                                         autograd::AutoGradCellImpl *auto_grad_cell_ptr) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  bool need_traverse = !auto_grad_cell_ptr->grad_by_value() && cnode->HasAttr(kIsKNode);
  if (need_traverse || cnode->seen_ == seen || IsPrimitiveCNode(cnode, prim::kPrimBpropCut) ||
      !IsPrimitiveCNode(cnode)) {
    return;
  }
  cnode->seen_ = seen;
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    ConvertMakeTupleInputToDynamicInput(cnode->input(kIndex1), seen, auto_grad_cell_ptr);
    return;
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    ConvertMakeTupleInputToDynamicInput(cnode->input(i), seen, auto_grad_cell_ptr);
  }

  if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) &&
      std::any_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                  [](const AnfNodePtr &node) { return node->abstract()->isa<abstract::AbstractSequence>(); })) {
    AnfNodePtrList plant_inputs;
    std::vector<int64_t> dyn_input_sizes;
    (void)plant_inputs.emplace_back(common::AnfAlgo::GetCNodePrimitiveNode(cnode));
    for (size_t i = 1; i < cnode->size(); ++i) {
      const auto &input_node = cnode->input(i);
      if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
        auto dyn_input_size = opt::SplitTupleInputs(auto_grad_cell_ptr->ad_param()->tape_, input_node, &plant_inputs);
        (void)dyn_input_sizes.emplace_back(dyn_input_size);
      } else {
        (void)plant_inputs.emplace_back(input_node);
        (void)dyn_input_sizes.emplace_back(-1);
      }
    }
    // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
    if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
      cnode->AddAttr(kTupleToMakeTuple, MakeValue(true));
      common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode);
      MS_LOG(DEBUG) << "Change node to dynamic len " << cnode->DebugString();
      cnode->set_inputs(plant_inputs);
      for (size_t i = 1; i < plant_inputs.size(); ++i) {
        auto_grad_cell_ptr->AddUser(plant_inputs[i], cnode, i);
      }
    }
  }
}

CNodePtr ConvertConstInputToAttr(const CNodePtr &cnode, const std::string &device_target, bool is_dynamic_shape,
                                 bool grad_by_value) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "Get cnode not primitive " << cnode->DebugString();
    return cnode;
  }

  mindspore::HashSet<size_t> input_to_attr = {};
  PyNativeAlgo::Common::GetConstInputToAttr(prim, prim->name(), device_target, is_dynamic_shape, &input_to_attr);
  if (input_to_attr.empty()) {
    TraverseCnode(cnode, device_target, is_dynamic_shape, grad_by_value);
    return cnode;
  }
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return cnode;
  }

  // Change to attr
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  AnfNodePtrList new_inputs{NewValueNode(prim)};
  size_t convert_size = 0;
  for (size_t i = 0; i < cnode->size() - 1; ++i) {
    auto input_node = cnode->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<ValueNode>() && input_to_attr.find(i) != input_to_attr.end()) {
      const auto &value_node = input_node->cast<ValueNodePtr>();
      MS_LOG(DEBUG) << "start erase input[" << i << "] of cnode[" + cnode->DebugString() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      const auto &value = value_node->value();
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          return cnode;
        }
      }
      ++convert_size;
      if (!grad_by_value) {
        auto &pair = node_attr_value_[cnode->ToString()];
        (void)pair.emplace_back(i, value);
      }
      prim->set_attr(input_names_vec[i], value);
    } else {
      (void)new_inputs.emplace_back(input_node);
    }
  }
  if (convert_size > 0) {
    cnode->AddAttr(kAttrConvertAttrNode, MakeValue(convert_size));
  }
  cnode->set_inputs(new_inputs);
  // If cast input has a cast
  TraverseCnode(cnode, device_target, is_dynamic_shape, grad_by_value);
  return cnode;
}

void ProcessAttrNode(const FuncGraphPtr &tape_graph, const CNodePtr &cnode, ValuePtrList *input_value,
                     AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->HasAttr(kAttrConvertAttrNode)) {
    const auto item = node_attr_value_.find(cnode->ToString());
    if (item != node_attr_value_.end()) {
      for (const auto &t : item->second) {
        (void)PyNativeAlgo::Common::SetValueGradInfo(t.second, nullptr, TensorGradType::kConstant);
        auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(t.second, nullptr);
        auto new_inputs = cnode->inputs();
        (void)new_inputs.insert(new_inputs.begin() + SizeToLong(t.first) + kIndex1, new_v_node);
        MS_EXCEPTION_IF_NULL(cnode_inputs);
        (void)cnode_inputs->insert(cnode_inputs->begin() + SizeToLong(t.first) + kIndex1, new_v_node);
        cnode->set_inputs(new_inputs);
        MS_EXCEPTION_IF_NULL(input_value);
        (void)input_value->insert(input_value->begin() + SizeToLong(t.first), t.second);
      }
      node_attr_value_.erase(item);
    }
  }
  RevertMakeTupleNode(tape_graph, cnode, input_value, cnode_inputs);
}

void ClearCache() { node_attr_value_.clear(); }
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
