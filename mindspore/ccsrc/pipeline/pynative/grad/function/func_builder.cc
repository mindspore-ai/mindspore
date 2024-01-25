/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/pynative/grad/function/func_builder.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "runtime/pynative/op_function/pyboost_grad_functions.h"
#include "pipeline/pynative/grad/function/function_utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore::pynative {
namespace {
ValuePtrList VectorRefToValuePtrList(const VectorRef &outputs) {
  ValuePtrList real_outputs;
  real_outputs.reserve(outputs.size());
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(real_outputs), [](const BaseRef &val) {
    auto value = utils::cast<ValuePtr>(val);
    MS_EXCEPTION_IF_NULL(value);
    return value;
  });
  return real_outputs;
}

AbstractBasePtr SetAbstractValueToAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kValueAny);
  } else if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    for (const auto &elem : abs_seq->elements()) {
      (void)SetAbstractValueToAnyValue(elem);
    }
  } else if (abs->isa<abstract::AbstractDictionary>()) {
    const auto &abs_dic = abs->cast<abstract::AbstractDictionaryPtr>();
    for (const auto &elem : abs_dic->elements()) {
      (void)SetAbstractValueToAnyValue(elem.first);
      (void)SetAbstractValueToAnyValue(elem.second);
    }
  }
  return abs;
}

void GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name, const std::string &device_target,
                         bool is_dynamic_shape, mindspore::HashSet<size_t> *input_to_attr_index) {
  if (op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    PrimitiveReadLock read_lock(op_prim->shared_mutex());
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_index);
    return;
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return;
  }

  auto reg_info =
    opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, device_target, is_dynamic_shape);
  if (reg_info == nullptr) {
    return;
  } else {
    MS_EXCEPTION_IF_NULL(input_to_attr_index);
    for (auto &iter : reg_info->input_attr_map()) {
      (void)input_to_attr_index->insert(iter.first);
    }
  }
}

NodePtrList ConvertConstInputToAttr(const PrimitivePtr &prim, const NodePtrList &inputs,
                                    const std::string &device_target) {
  mindspore::HashSet<size_t> input_to_attr;
  GetConstInputToAttr(prim, prim->name(), device_target, false, &input_to_attr);
  if (input_to_attr.empty()) {
    return inputs;
  }
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return inputs;
  }
  NodePtrList real_inputs;
  real_inputs.reserve(inputs.size());
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &value = inputs[i]->Value();
    if (input_to_attr.find(i) != input_to_attr.end()) {
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          return inputs;
        }
      }
      prim->set_attr(input_names_vec[i], value);
    } else {
      (void)real_inputs.emplace_back(inputs[i]);
    }
  }
  return real_inputs;
}
}  // namespace
NodePtr FuncBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  auto real_inputs = ConvertConstInputToAttr(prim, inputs, device_target_);
  std::vector<ValuePtr> op_inputs;
  op_inputs.reserve(real_inputs.size());
  abstract::AbstractBasePtrList input_abs;
  input_abs.reserve(real_inputs.size());
  std::vector<InputType> input_mask;
  input_mask.reserve(real_inputs.size());
  for (const auto &input : real_inputs) {
    (void)op_inputs.emplace_back(input->Value());
    auto abs = input->abstract();
    if (input->input_type() != InputType::kConstant) {
      (void)SetAbstractValueToAnyValue(abs);
    }
    (void)input_abs.emplace_back(abs);
    (void)input_mask.emplace_back(input->input_type());
  }
  VectorRef outputs;
  kernel::pyboost::OpRunnerInfo op_runner_info{.prim = prim,
                                               .device_target = device_target_,
                                               .inputs = op_inputs,
                                               .inputs_abs = input_abs,
                                               .inputs_mask = input_mask,
                                               .output_abs = nullptr};

  runtime::PyBoostOpExecute::GetInstance().Execute(&op_runner_info, &outputs);
  auto real_outputs = VectorRefToValuePtrList(outputs);
  ValuePtr value_result;
  if (real_outputs.size() != 1) {
    value_result = std::make_shared<ValueTuple>(std::move(real_outputs));
  } else {
    value_result = real_outputs[kIndex0];
  }
  auto result = NewFuncNode(value_result, InputType::kOpOutput);
  if (op_runner_info.output_abs != nullptr) {
    result->set_abstract(op_runner_info.output_abs);
  }
  return result;
}

NodePtr FuncBuilder::EmitValue(const ValuePtr &value) {
  auto node = NewFuncNode(value, InputType::kConstant);
  return node;
}

NodePtr FuncBuilder::Concat(const NodePtr &inputs, int64_t axis, const NodePtr &out) {
  NodePtrList node_inputs = FlattenNode(inputs, out);
  return Concat(node_inputs, axis);
}

NodePtr FuncBuilder::Concat(const NodePtrList &inputs, int64_t axis) {
  std::vector<int64_t> dyn_size{static_cast<int64_t>(inputs.size()), -1};
  expander::DAttr attrs{std::make_pair(kAttrDynInputSizes, MakeValue(dyn_size))};
  NodePtrList real_input(inputs);
  (void)real_input.emplace_back(Value(axis));
  return Emit(kConcatOpName, real_input, attrs);
}

NodePtr FuncBuilder::Stack(const NodePtr &x, const ValuePtr &axis_value, const NodePtr &out) {
  NodePtrList node_inputs = FlattenNode(x, out);
  int64_t axis = GetValue<int64_t>(axis_value);
  return Stack(node_inputs, axis);
}

NodePtr FuncBuilder::Stack(const NodePtrList &x, int64_t axis) {
  std::vector<int64_t> dyn_size{static_cast<int64_t>(x.size()), -1};
  expander::DAttr attrs{std::make_pair(kAttrDynInputSizes, MakeValue(dyn_size)),
                        std::make_pair("axis", MakeValue(axis))};
  return Emit(kStackOpName, x, attrs);
}

NodePtr FuncBuilder::Cast(const NodePtr &node, const TypePtr &type) {
  auto prim = NewPrimitive(kCastOpName);
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
  }
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  if (input_names_vec.size() != kSizeTwo) {
    MS_LOG(EXCEPTION) << "Cast input names size should be 2, but got " << input_names_vec.size();
  }
  prim->set_attr(input_names_vec[kIndex1], type);
  return EmitOp(prim, {node});
}

NodePtr FuncBuilder::BatchNormGrad(const NodePtrList &inputs) {
  if (device_target_ != kAscendDevice) {
    return Emitter::BatchNormGrad(inputs);
  }
  auto prim = NewPrimitive("BNInferGrad");
  NodePtrList real_inputs;
  real_inputs.reserve(kSizeFour);
  constexpr size_t kIdxGrads = 0;
  constexpr size_t kIdxScale = 2;
  constexpr size_t kIdxVariance = 4;
  constexpr size_t kIdxIsTraining = 6;
  constexpr size_t kIdxEpsilon = 7;
  prim->set_attr(kAttrIsTraining, inputs[kIdxIsTraining]->Value());
  prim->set_attr(kAttrEpsilon, inputs[kIdxEpsilon]->Value());
  real_inputs.emplace_back(inputs[kIdxGrads]);
  real_inputs.emplace_back(inputs[kIdxScale]);
  real_inputs.emplace_back(inputs[kIdxVariance]);
  real_inputs.emplace_back(inputs[kIdxEpsilon]);
  return Emit("BNInferGrad", real_inputs);
}

NodePtr FuncBuilder::SparseSoftmaxCrossEntropyWithLogits(const NodePtr &logits, const NodePtr &labels, bool is_grad) {
  return Emitter::SparseSoftmaxCrossEntropyWithLogits(logits, labels, is_grad);
  //  if (device_target_ != kAscendDevice || !is_grad) {
  //  }
  //  Reshape();
  //  auto onehot = Emit("OneHot", {out_0, depth, on_value, off_value, ib->Value<int64_t>(onehot_axis)})
}

NodePtr FuncBuilder::TupleGetItem(const NodePtr &input, size_t i) {
  auto value = input->Value();
  if (!value->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Input value should be sequence"
                      << "but got " << value->ToString();
  }
  auto seq = value->cast<ValueSequencePtr>();
  if (seq->size() <= i) {
    MS_LOG(EXCEPTION) << "Input value sequence size should > " << i << " but got " << value->ToString();
  }
  return NewFuncNode(seq->value()[i], input->input_type());
}

NodePtr FuncBuilder::OutZeros(const NodePtr &node) { return NewFuncNode(kNone, InputType::kConstant); }

ValuePtr FuncBuilder::Ones(const ValuePtr &value) {
  NodePtrList inputs{NewFuncNode(value, InputType::kOpOutput)};
  return EmitOp(prim::kPrimOnesLike, inputs)->Value();
}

ValuePtr FuncBuilder::Zeros(const ValuePtr &value) {
  auto input = NewFuncNode(value, InputType::kInput);
  return ZerosLike(input)->Value();
}

NodePtr FuncBuilder::TupleGetItem(const NodePtr &input, const NodePtr &index) {
  auto value = index->Value();
  size_t i = GetValue<int64_t>(value);
  return TupleGetItem(input, i);
}

NodePtr FuncBuilder::MakeTuple(const NodePtrList &inputs) {
  ValuePtrList values;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(values),
                 [](const NodePtr &node) { return node->Value(); });
  auto value = std::make_shared<ValueTuple>(values);
  auto tuple_node = NewFuncNode(value, InputType::kOpOutput);
  return tuple_node;
}

void FuncBuilder::SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                            mindspore::HashMap<std::string, ValuePtr> *attrs_ptr) {
  instance_name_ = std::move(instance_name);
  inputs_ptr_ = inputs;
  attrs_ptr_ = attrs_ptr;
}

NodePtrList FuncBuilder::FlattenNode(const NodePtr &input, const NodePtr &out) {
  if (!input->Value()->isa<ValueSequence>()) {
    return {input};
  }
  auto value_seq = input->Value()->cast<ValueSequencePtr>()->value();
  auto out_seq = out->Value()->cast<ValueSequencePtr>()->value();
  NodePtrList flattenNodes;
  flattenNodes.reserve(value_seq.size());
  for (size_t i = 0; i < value_seq.size(); ++i) {
    auto value = value_seq[i];
    if (value->isa<None>()) {
      (void)flattenNodes.emplace_back(ZerosLike(NewFuncNode(out_seq[i], out->input_type())));
      continue;
    }
    (void)flattenNodes.emplace_back(NewFuncNode(value, input->input_type()));
  }
  return flattenNodes;
}
}  // namespace mindspore::pynative
