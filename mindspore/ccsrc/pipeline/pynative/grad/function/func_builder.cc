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
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore::pynative::autograd {
namespace {
template <typename T>
std::string DebugInput(std::vector<T> items) {
  static constexpr size_t end_char_size = 2;
  std::ostringstream buf;
  for (size_t i = 0; i < items.size(); ++i) {
    if (items[i]->template isa<tensor::Tensor>()) {
      auto tensor = items[i]->template cast<tensor::TensorPtr>();
      auto grad = std::make_shared<tensor::Tensor>(*tensor);
      grad->data_sync();
      buf << i << "th: "
          << "ptr " << items[i].get() << ", " << grad->ToStringRepr() << ", ";
    } else {
      buf << i << "th: "
          << "ptr " << items[i].get() << ", " << items[i]->ToString() << ", ";
    }
  }
  return buf.str().erase(buf.str().size() - end_char_size);
}

std::set<int64_t> GetValueDependArgIndices(const PrimitivePtr &primitive, const NodePtrList &inputs) {
  auto depend_list = ops::GetInputDependValueList(primitive);
  auto attr = primitive->GetAttr(kAttrDynInputSizes);
  if (attr == nullptr) {
    return depend_list;
  }
  // mapping from input prototype index to corresponding start index of real input
  std::vector<int64_t> dyn_input_sizes = GetValue<std::vector<int64_t>>(attr);
  if (!dyn_input_sizes.empty()) {
    auto temp_depend_list = depend_list;
    depend_list.clear();
    for (const auto item : temp_depend_list) {
      int64_t offset = 0;
      for (int64_t i = 0; i < item; i++) {
        auto idx = static_cast<size_t>(i);
        if (dyn_input_sizes[idx] == -1) {
          offset += 1;
        } else {
          offset += dyn_input_sizes[idx];
        }
      }
      depend_list.emplace(offset);
      MS_LOG(DEBUG) << "Adjust depend list from " << item << " to " << offset << " for op: " << primitive->name();
    }
  }
  return depend_list;
}

void SetDependValue(const PrimitivePtr &primitive, const NodePtrList &inputs) {
  auto depend_list = GetValueDependArgIndices(primitive, inputs);
  if (depend_list.empty()) {
    return;
  }
  int64_t input_size = inputs.size();
  for (const auto index : depend_list) {
    if (index >= input_size) {
      MS_LOG(EXCEPTION) << "For depend list index should be less than inputs size: " << input_size
                        << ", but got index: " << index;
    }
    const auto abstract = inputs[index]->abstract();
    const auto value = inputs[index]->Value();
    abstract->set_value(value);
  }
}
}  // namespace

NodePtr FuncBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_LOG(DEBUG) << "Emit op " << prim->name();
  auto real_inputs = pass_forward_->PassForOpInput(prim, inputs);
  std::vector<ValuePtr> op_inputs;
  op_inputs.reserve(real_inputs.size());
  abstract::AbstractBasePtrList input_abs;
  input_abs.reserve(real_inputs.size());
  std::vector<InputType> input_mask;
  input_mask.reserve(real_inputs.size());
  SetDependValue(prim, inputs);
  for (const auto &input : real_inputs) {
    auto value = input->Value();
    auto abs = input->abstract();
    if (value->isa<None>()) {
      if (!abs->isa<abstract::AbstractNone>()) {
        auto out_tensor = std::make_shared<tensor::Tensor>(input->dtype()->type_id(), input->shape());
        auto zero_node = ZerosLike(NewFuncNode(out_tensor, abs, input->input_type()));
        value = zero_node->Value();
      } else {
        MS_LOG(DEBUG) << "None value abstract got None abstract!";
      }
    }
    (void)op_inputs.emplace_back(value);
    (void)input_abs.emplace_back(abs);
    (void)input_mask.emplace_back(input->input_type());
  }
  MS_LOG(DEBUG) << "Get input value size " << op_inputs.size() << ", " << DebugInput<ValuePtr>(op_inputs);
  MS_LOG(DEBUG) << "Get input abs size " << input_abs.size() << ", "
                << DebugInput<abstract::AbstractBasePtr>(input_abs);
  VectorRef outputs;
  kernel::pyboost::OpRunnerInfo op_runner_info{.prim = prim,
                                               .device_target = device_target_,
                                               .inputs = op_inputs,
                                               .inputs_abs = input_abs,
                                               .inputs_mask = input_mask,
                                               .output_abs = nullptr};
  runtime::PyBoostOpExecute::GetInstance().Execute(&op_runner_info, &outputs);
  auto real_outputs = PyNativeAlgo::DataConvert::VectorRefToValuePtrList(outputs);
  MS_LOG(DEBUG) << "Get output value size " << real_outputs.size() << ", " << DebugInput<ValuePtr>(real_outputs);
  ValuePtr value_result;
  if (real_outputs.size() != 1) {
    value_result = std::make_shared<ValueTuple>(std::move(real_outputs));
  } else {
    value_result = real_outputs[kIndex0];
  }
  MS_EXCEPTION_IF_NULL(op_runner_info.output_abs);
  auto result = NewFuncNode(value_result, op_runner_info.output_abs, InputType::kOpOutput);
  return result;
}

NodePtr FuncBuilder::EmitValue(const ValuePtr &value) {
  // For constant value, its abstract may not use, we delay set abs, if op use its abstract, we can get abstract
  // from FuncBuilder::abstract()
  auto node = NewFuncNode(value, nullptr, InputType::kConstant);
  return node;
}

NodePtr FuncBuilder::Concat(const NodePtr &inputs, int64_t axis) {
  NodePtrList node_inputs = FlattenNode(inputs);
  return Concat(node_inputs, axis);
}

NodePtr FuncBuilder::Concat(const NodePtrList &inputs, int64_t axis) {
  std::vector<int64_t> dyn_size{static_cast<int64_t>(inputs.size()), -1};
  expander::DAttr attrs{std::make_pair(kAttrDynInputSizes, MakeValue(dyn_size))};
  NodePtrList real_input(inputs);
  (void)real_input.emplace_back(Value(axis));
  return Emit(kConcatOpName, real_input, attrs);
}

NodePtr FuncBuilder::Stack(const NodePtr &x, const ValuePtr &axis_value) {
  NodePtrList node_inputs = FlattenNode(x);
  int64_t axis = GetValue<int64_t>(axis_value);
  return Stack(node_inputs, axis);
}

NodePtr FuncBuilder::Stack(const NodePtrList &x, int64_t axis) {
  std::vector<int64_t> dyn_size{static_cast<int64_t>(x.size()), -1};
  expander::DAttr attrs{std::make_pair(kAttrDynInputSizes, MakeValue(dyn_size)),
                        std::make_pair("axis", MakeValue(axis))};
  return Emit(kStackOpName, x, attrs);
}

NodePtr FuncBuilder::BatchNormGrad(const NodePtrList &inputs) {
  return pass_forward_->BatchNormGradToBNInferGrad(inputs);
}

NodePtr FuncBuilder::SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const expander::DAttr &attrs,
                                                         const NodePtr &out, const NodePtr &dout, bool is_graph_mode) {
  return pass_forward_->GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(inputs, attrs, out, dout, is_graph_mode);
}

NodePtr FuncBuilder::Depend(const NodePtr &value, const NodePtr &expr) { return value; }

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
  abstract::AbstractBasePtr item_abs = nullptr;
  auto seq_abs = input->abstract()->cast<abstract::AbstractSequencePtr>();
  if (seq_abs != nullptr && seq_abs->size() == seq->size()) {
    item_abs = seq_abs->elements()[i];
  }
  return NewFuncNode(seq->value()[i], item_abs, input->input_type());
}

NodePtr FuncBuilder::OutZeros(const NodePtr &node) { return NewFuncNode(kNone, nullptr, InputType::kConstant); }

ValuePtr FuncBuilder::Ones(const ValuePtr &value) {
  auto ones_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract());
  NodePtrList inputs{NewFuncNode(value, ones_abs, InputType::kOpOutput)};
  return EmitOp(prim::kPrimOnesLike, inputs)->Value();
}

ValuePtr FuncBuilder::Zeros(const ValuePtr &value) {
  auto zeros_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract());
  auto input = NewFuncNode(value, zeros_abs, InputType::kOpOutput);
  return ZerosLike(input)->Value();
}

ValuePtr FuncBuilder::Add(const ValuePtr &input, const ValuePtr &other) {
  auto input_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(input->ToAbstract());
  auto other_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(other->ToAbstract());
  auto input_node = NewFuncNode(input, input_abs, InputType::kOpOutput);
  auto other_node = NewFuncNode(other, other_abs, InputType::kOpOutput);
  return Emit(mindspore::kAddOpName, {input_node, other_node})->Value();
}

NodePtr FuncBuilder::TupleGetItem(const NodePtr &input, const NodePtr &index) {
  auto value = index->Value();
  size_t i = GetValue<int64_t>(value);
  return TupleGetItem(input, i);
}

NodePtr FuncBuilder::MakeTuple(const NodePtrList &inputs) {
  ValuePtrList values;
  AbstractBasePtrList abs;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(values),
                 [](const NodePtr &node) { return node->Value(); });
  auto value = std::make_shared<ValueTuple>(values);
  auto tuple_node = NewFuncNode(value, nullptr, InputType::kOpOutput);
  return tuple_node;
}

NodePtr FuncBuilder::MakeList(const NodePtrList &inputs) { return MakeTuple(inputs); }

void FuncBuilder::SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                            mindspore::HashMap<std::string, ValuePtr> *attrs_ptr) {
  instance_name_ = std::move(instance_name);
  inputs_ptr_ = inputs;
  attrs_ptr_ = attrs_ptr;
}

NodePtrList FuncBuilder::FlattenNode(const NodePtr &input) {
  if (!input->Value()->isa<ValueSequence>()) {
    return {input};
  }
  auto value_seq = input->Value()->cast<ValueSequencePtr>()->value();
  auto value_abs = input->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_abs);
  NodePtrList flattenNodes;
  flattenNodes.reserve(value_seq.size());
  for (size_t i = 0; i < value_seq.size(); ++i) {
    auto value = value_seq[i];
    (void)flattenNodes.emplace_back(NewFuncNode(value, value_abs->elements()[i], input->input_type()));
  }
  return flattenNodes;
}
}  // namespace mindspore::pynative::autograd
