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
#include <set>
#include "runtime/pynative/op_function/pyboost_grad_functions.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/core/ops/op_utils.h"
#include "frontend/operator/cc_implementations.h"
#include "mindspore/ccsrc/kernel/pyboost/op_register.h"
#include "kernel/pyboost/auto_generate/cast.h"

namespace mindspore::pynative::autograd {
namespace {
template <typename T>
std::string PrintDebugInfo(std::vector<T> items, const std::string &info_header = "") {
  static constexpr size_t end_char_size = 2;
  std::ostringstream buf;
  buf << info_header;
  for (size_t i = 0; i < items.size(); ++i) {
    if (items[i] == nullptr) {
      MS_LOG(DEBUG) << "The " << i << "'th item is nullptr!";
      continue;
    }
    if (items[i]->template isa<tensor::BaseTensor>()) {
      auto tensor = items[i]->template cast<tensor::BaseTensorPtr>();
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
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    if (tensor != nullptr) {
      tensor->data_sync();
    }
    abstract->set_value(value);
  }
}

std::vector<int64_t> BuildShape(const abstract::AbstractBasePtr &abs) {
  auto base_shape = abs->BuildShape();
  if (base_shape->isa<abstract::NoShape>()) {
    return {};
  }
  auto shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape->shape();
}

bool ParseCond(const NodePtr &cond) {
  auto cond_val = cond->Value();
  if (cond_val->isa<BoolImm>()) {
    return GetValue<bool>(cond_val);
  }
  if (cond_val->isa<tensor::BaseTensor>()) {
    auto tensor = cond_val->cast<tensor::BaseTensorPtr>();
    tensor->data_sync();
    size_t data_size = tensor->DataSize();
    auto tensor_type = tensor->Dtype();
    if (tensor_type->type_id() == kNumberTypeBool) {
      auto data_c = reinterpret_cast<bool *>(tensor->data_c());
      MS_EXCEPTION_IF_NULL(data_c);
      return std::all_of(data_c, data_c + data_size, [](const bool &data) { return static_cast<bool>(data); });
    }
  }
  MS_LOG(EXCEPTION) << "For control flow, the cond should be Tensor[bool] or bool, but got: " << cond_val->ToString();
}
}  // namespace

NodePtr FuncBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kEmitOp, prim->name(),
                                     false);
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
    auto abs = input->abstract();
    auto value = FillZeros(input->Value(), abs);
    (void)op_inputs.emplace_back(value);
    (void)input_abs.emplace_back(abs);
    (void)input_mask.emplace_back(input->input_type());
  }
  MS_LOG(DEBUG) << "Get input value size " << op_inputs.size() << ", "
                << PyNativeAlgo::Common::PrintDebugInfo(op_inputs);
  MS_LOG(DEBUG) << "Get input abs size " << input_abs.size() << ", " << PyNativeAlgo::Common::PrintDebugInfo(input_abs);
  VectorRef outputs;
  runtime::OpRunnerInfo op_runner_info{prim, device_target_, op_inputs, input_abs, input_mask, nullptr};
  runtime::PyBoostOpExecute::GetInstance().Execute(&op_runner_info, &outputs);
  auto real_outputs = common::AnfAlgo::TransformVectorRefToMultiValue(outputs);
  MS_LOG(DEBUG) << "Get output value size " << real_outputs.size() << ", "
                << PyNativeAlgo::Common::PrintDebugInfo(real_outputs);
  if (op_runner_info.output_value_simple_info != nullptr) {
    // Get output abstract
    op_runner_info.output_abs = TransformValueSimpleInfoToAbstract(*op_runner_info.output_value_simple_info);
  }
  ValuePtr value_result;
  MS_EXCEPTION_IF_NULL(op_runner_info.output_abs);
  if (real_outputs.size() == kSizeOne && !op_runner_info.output_abs->isa<abstract::AbstractSequence>()) {
    value_result = real_outputs[kIndex0];
  } else {
    value_result = std::make_shared<ValueTuple>(std::move(real_outputs));
  }
  // Set abstract to tensor cache
  if (op_runner_info.output_value_simple_info != nullptr) {
    PyNativeAlgo::AutoGrad::CacheOutputAbstract(value_result, op_runner_info.output_abs);
  }
  auto result = NewFuncNode(value_result, op_runner_info.output_abs, InputType::kOpOutput);
  return result;
}

NodePtr FuncBuilder::EmitValue(const ValuePtr &value) {
  // For constant value, its abstract may not use, we delay set abs, if op use its abstract, we can get abstract
  // from FuncBuilder::abstract()
  auto node = NewFuncNode(value, nullptr, InputType::kConstant);
  return node;
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

NodePtr FuncBuilder::BatchNormGrad(const NodePtrList &inputs, bool is_scale_or_bias_grad) {
  return pass_forward_->BatchNormGradToBNInferGrad(inputs, is_scale_or_bias_grad);
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

NodePtr FuncBuilder::OutZeros(const NodePtr &node) {
  if (!node->Value()->isa<ValueSequence>()) {
    return NewFuncNode(kNone, nullptr, InputType::kConstant);
  }
  auto val_seq = node->Value()->cast<ValueSequencePtr>();
  if (val_seq->size() == kSizeZero) {
    return NewFuncNode(kNone, nullptr, InputType::kConstant);
  }
  const auto &value = val_seq->value()[kIndexZero];
  if (!value->isa<tensor::Tensor>()) {
    return NewFuncNode(kNone, nullptr, InputType::kConstant);
  } else {
    ValuePtrList values(val_seq->size(), kNone);
    return NewFuncNode(std::make_shared<ValueTuple>(values), nullptr, InputType::kConstant);
  }
}

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
  return EmitOp(prim::kPrimAdd, {input_node, other_node})->Value();
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

NodePtr FuncBuilder::Conditional(const NodePtr &cond, const expander::Emitter::BlockFunc &true_case,
                                 const expander::Emitter::BlockFunc &false_case) {
  NodePtrList result;
  if (ParseCond(cond)) {
    result = true_case(this);
  } else {
    result = false_case(this);
  }
  if (result.size() == kSizeOne) {
    return result[kIndex0];
  }
  return MakeTuple(result);
}

NodePtr FuncBuilder::ScalarEq(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto lhs_val = lhs->Value();
  auto rhs_val = rhs->Value();
  ValuePtr result;
  if (lhs_val->isa<BoolImm>() && rhs_val->isa<BoolImm>()) {
    result = MakeValue(GetValue<bool>(lhs_val) == GetValue<bool>(rhs_val));
  } else {
    result = prim::ScalarEq({lhs->Value(), rhs->Value()});
  }
  MS_LOG(DEBUG) << "ScalarEq op: lhs " << lhs_val->ToString() << ", rhs " << rhs_val->ToString();
  return NewFuncNode(result, nullptr, InputType::kOpOutput);
}

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
    auto &value = value_seq[i];
    (void)flattenNodes.emplace_back(NewFuncNode(value, value_abs->elements()[i], input->input_type()));
  }
  return flattenNodes;
}

ValuePtr FuncBuilder::FillZeros(const ValuePtr &value, const abstract::AbstractBasePtr &abs) {
  auto convert_value = value;
  if (value->isa<None>()) {
    if (abs->isa<abstract::AbstractTensor>()) {
      auto tensor_dtype = abs->BuildType()->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_dtype);
      auto dtype = tensor_dtype->element();
      auto shape = BuildShape(abs);
      auto out_tensor = std::make_shared<tensor::Tensor>(dtype->type_id(), shape);
      auto zero_node = ZerosLike(NewFuncNode(out_tensor, abs, InputType::kOpOutput));
      convert_value = zero_node->Value();
    } else {
      MS_LOG(DEBUG) << "None value abstract got None abstract!";
    }
  } else if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    auto abs_list = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_list);
    std::vector<ValuePtr> value_list;
    value_list.reserve(seq->value().size());
    for (size_t i = 0; i < seq->value().size(); ++i) {
      const auto &val = seq->value()[i];
      const auto &temp_abs = abs_list->elements()[i];
      auto convert = FillZeros(val, temp_abs);
      (void)value_list.emplace_back(convert);
    }
    convert_value = std::make_shared<ValueTuple>(value_list);
  }
  return convert_value;
}
}  // namespace mindspore::pynative::autograd
