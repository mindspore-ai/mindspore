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

namespace mindspore::pynative::autograd {
namespace {
template <typename T>
std::string DebugInput(std::vector<T> items) {
  static constexpr size_t end_char_size = 2;
  std::ostringstream buf;
  for (size_t i = 0; i < items.size(); ++i) {
    buf << i << "th: "
        << "ptr " << items[i].get() << ", " << items[i]->ToString() << ", ";
  }
  return buf.str().erase(buf.str().size() - end_char_size);
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
  for (const auto &input : real_inputs) {
    (void)op_inputs.emplace_back(input->Value());
    auto abs = input->abstract();
    if (input->input_type() != InputType::kConstant) {
      (void)PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs);
    }
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
  return pass_forward_->BatchNormGradToBNInferGrad(inputs);
}

NodePtr FuncBuilder::SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const expander::DAttr &attrs,
                                                         const NodePtr &out, const NodePtr &dout, bool is_graph_mode) {
  return pass_forward_->GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(inputs, attrs, out, dout, is_graph_mode);
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
  auto input = NewFuncNode(value, InputType::kOpOutput);
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
}  // namespace mindspore::pynative::autograd
