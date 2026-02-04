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

#include "pynative/backward/op_grad/func_builder.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include "ir/dtype/tensor_type.h"
#include "include/pynative/utils/pyboost/grad_functions/pyboost_grad_functions.h"
#include "include/pynative/utils/pyboost/comm_handle.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/pass_manager/op_adaptation_info_factory.h"
#include "include/utils/pynative/common_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "pynative/backward/grad_utils.h"
#include "frontend/operator/cc_implementations.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "pynative/backward/op_grad/auto_generate/pyboost_native_grad_functions.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"

namespace mindspore::pynative::autograd {
namespace {
void FlattenShape(const NodePtr &input, ShapeArray *args, std::vector<std::vector<size_t>> *pos_idx) {
  MS_EXCEPTION_IF_NULL(input);
  // input[i]'s shape is used
  const auto &abs = input->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractSequence>()) {
    auto input_shape = input->shape();
    pos_idx->push_back({args->size()});
    (void)args->emplace_back(input_shape);
  } else {
    const auto &sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
    (void)ops::TryGetShapeArg(sequence_abs, args, pos_idx);
  }
}

template <typename T>
std::vector<T> ConvertValueSeqToVector(const ValueSequencePtr &tuple) {
  const auto &values = tuple->value();
  std::vector<T> result;
  result.reserve(values.size());
  for (const auto &value : values) {
    (void)result.emplace_back(GetValue<T>(value));
  }
  MS_LOG(DEBUG) << "Convert ValueTuple to vector " << result;
  return result;
}

std::vector<int64_t> GetIntList(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value_ptr = node->BuildValue();
  if (value_ptr->isa<ValueSequence>()) {
    const auto &seq = value_ptr->cast<ValueSequencePtr>();
    return ConvertValueSeqToVector<int64_t>(seq);
  }
  if (value_ptr->isa<Int64Imm>()) {
    return {GetValue<int64_t>(value_ptr)};
  }
  if (value_ptr->isa<Int32Imm>()) {
    return {static_cast<int64_t>(GetValue<int64_t>(value_ptr))};
  }
  if (value_ptr->isa<tensor::Tensor>()) {
    auto tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    // In pynative mode, need data sync before get tensor value, otherwise the tensor value may be undefined.
    auto cpu_tensor = tensor->cpu();
    return CheckAndConvertUtils::CheckTensorIntValue("value", cpu_tensor, "GetIntList");
  }
  return std::vector<int64_t>{};
}

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
    if (items[i]->template isa<tensor::Tensor>()) {
      auto tensor = items[i]->template cast<tensor::TensorPtr>();
      auto grad = std::make_shared<tensor::Tensor>(*tensor);
      auto cpu_grad = grad->cpu();
      buf << i << "th: "
          << "ptr " << items[i].get() << ", " << cpu_grad->ToStringRepr() << ", ";
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
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor != nullptr) {
      auto cpu_tensor = tensor->cpu();
      inputs[index]->SetValue(cpu_tensor);
      abstract->set_value(cpu_tensor);
      continue;
    }
    abstract->set_value(value);
  }
}

bool ParseCond(const NodePtr &cond) {
  MS_EXCEPTION_IF_NULL(cond);
  auto cond_val = cond->Value();
  if (cond_val->isa<BoolImm>()) {
    return GetValue<bool>(cond_val);
  }
  if (cond_val->isa<tensor::Tensor>()) {
    auto tensor = cond_val->cast<tensor::TensorPtr>();
    auto cpu_tensor = tensor->cpu();
    size_t data_size = cpu_tensor->DataSize();
    auto tensor_type = cpu_tensor->Dtype();
    if (tensor_type->type_id() == kNumberTypeBool) {
      auto data_c = reinterpret_cast<bool *>(cpu_tensor->data_c());
      MS_EXCEPTION_IF_NULL(data_c);
      return std::all_of(data_c, data_c + data_size, [](const bool &data) { return static_cast<bool>(data); });
    }
  }
  MS_LOG(EXCEPTION) << "For control flow, the cond should be Tensor[bool] or bool, but got: " << cond_val->ToString();
}

template <typename T, typename U>
bool InnerScalarEq(T x, U y) {
  if (std::isinf(static_cast<double>(x)) && std::isinf(static_cast<double>(y))) {
    return (x > 0 && y > 0) || (x < 0 && y < 0);
  }
  double error = static_cast<double>(x) - static_cast<double>(y);
  error = fabs(error);
  return error < DBL_EPSILON;
}

ValuePtr ScalarEqual(const ValuePtr &x, const ValuePtr &y) {
  MS_EXCEPTION_IF_NULL(y);
  if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {
    MS_EXCEPTION_IF_NULL(x);
    bool sum = InnerScalarEq(GetValue<float>(x), GetValue<float>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {
    bool sum = InnerScalarEq(GetValue<int>(x), GetValue<int>(y));
    return MakeValue(sum);
  }
  if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {
    bool sum = InnerScalarEq(GetValue<float>(x), GetValue<int>(y));
    return MakeValue(sum);
  }
  if (x->isa<FP32Imm>() && y->isa<Int64Imm>()) {
    bool sum = InnerScalarEq(GetValue<float>(x), GetValue<int64_t>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {
    bool sum = InnerScalarEq(GetValue<int>(x), GetValue<float>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int64Imm>() && y->isa<FP32Imm>()) {
    bool sum = InnerScalarEq(GetValue<int64_t>(x), GetValue<float>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {
    bool sum = InnerScalarEq(GetValue<int64_t>(x), GetValue<int64_t>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {
    bool sum = InnerScalarEq(GetValue<int64_t>(x), GetValue<int>(y));
    return MakeValue(sum);
  }
  if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {
    bool sum = InnerScalarEq(GetValue<int>(x), GetValue<int64_t>(y));
    return MakeValue(sum);
  }
  MS_EXCEPTION(TypeError) << "Unsupported input type for Scalar" << ", type of x:" << x->type_name()
                          << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()
                          << ", value of y:" << y->ToString();
}
}  // namespace

FuncBuilder::FuncBuilder(const std::string &name, device::DeviceType device_target,
                         const expander::ExpanderInferPtr &infer)
    : BpropBuilder(name, infer), device_target_(device_target) {
  pass_forward_ = std::make_shared<bprop_pass::FuncPassForward>(this, device_target);
  NativeFunc::set_device_target(device_target_);
}

NodePtr FuncBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(prim);
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
    AutoGradUtil::CacheOutputAbstract(value_result, op_runner_info.output_abs);
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

NodePtr FuncBuilder::Shape(const NodePtr &node, bool tensor) {
  auto shape = node->shape();
  if (tensor) {
    return Tensor(shape);
  } else {
    return Value(shape);
  }
}

void FuncBuilder::MarkSharedGradTensor(const NodePtr &lhs, const NodePtr &rhs) {
  if (lhs.get() == rhs.get()) {
    auto tensor = lhs->Value()->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_user_data("kSharedGradTensor", std::make_shared<bool>(true));
  }
}

NodePtrList FuncBuilder::ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs) {
  size_t input_size = inputs.size();
  ShapeArray const_args;
  const_args.reserve(input_size);
  std::vector<std::vector<size_t>> pos_idx;
  pos_idx.reserve(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    FlattenShape(inputs[i], &const_args, &pos_idx);
  }
  NodePtrList res;
  auto out = functor->Calc(const_args, pos_idx);
  res.reserve(out.size());
  (void)std::transform(out.begin(), out.end(), std::back_inserter(res),
                       [this](const ShapeVector &sh) { return Value(sh); });
  return res;
}

NodePtrList FuncBuilder::ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs,
                                   const std::vector<int64_t> &value_depend) {
  std::vector<bool> only_depend_shape(inputs.size(), true);
  for (auto idx : value_depend) {
    only_depend_shape[LongToSize(idx)] = false;
  }
  size_t input_size = inputs.size();
  ShapeArray const_args;
  const_args.reserve(input_size);
  std::vector<std::vector<size_t>> pos_idx;
  pos_idx.reserve(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    if (!only_depend_shape[i]) {
      // input[i]'s value is used
      const auto shape = GetIntList(inputs[i]);
      pos_idx.push_back({const_args.size()});
      const_args.push_back(shape);
    } else {
      FlattenShape(inputs[i], &const_args, &pos_idx);
    }
  }
  NodePtrList res;
  auto out = functor->Calc(const_args, pos_idx);
  res.reserve(out.size());
  (void)std::transform(out.begin(), out.end(), std::back_inserter(res),
                       [this](const ShapeVector &sh) { return Value(sh); });
  return res;
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

NodePtr FuncBuilder::Cast(const NodePtr &node, const TypePtr &type) {
  if (node->dtype()->type_id() == type->type_id()) {
    return node;
  }
  return NativeFunc::Cast(node, Value(static_cast<int64_t>(type->type_id())));
}

NodePtr FuncBuilder::Reshape(const NodePtr &node, const NodePtr &shape) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(shape->Value());
  if (!shape->Value()->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Reshape op second input should be vector<int> "
                      << "but got" << shape->Value()->ToString();
  }
  const auto &seq = shape->Value()->cast<ValueSequencePtr>();
  auto dst_shape = ConvertValueSeqToVector<int64_t>(seq);
  auto node_shape = node->shape();
  if (node_shape == dst_shape) {
    return node;
  }
  return NativeFunc::Reshape(node, shape);
}

NodePtr FuncBuilder::Transpose(const NodePtr &node, const NodePtr &perm) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(perm);
  MS_EXCEPTION_IF_NULL(perm->Value());
  if (!perm->Value()->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Transpose op second input should be vector<int> "
                      << "but got" << perm->Value()->ToString();
  }
  const auto &seq = perm->Value()->cast<ValueSequencePtr>();
  auto perm_list = ConvertValueSeqToVector<int64_t>(seq);
  // perm like [0, 1, 2, 3] does not need transpose.
  auto n = SizeToLong(perm_list.size());
  for (size_t i = 0; i < perm_list.size(); ++i) {
    // perm value may be negative, e.g. [0, -3, 2, 3] is equal to [0, 1, 2, 3]
    auto perm_i = perm_list[i] < 0 ? (perm_list[i] + n) : perm_list[i];
    if (perm_i != static_cast<int64_t>(i)) {
      return NativeFunc::Transpose(node, perm);
    }
  }
  return node;
}

NodePtr FuncBuilder::BroadcastTo(const NodePtr &x, const NodePtr &y) {
  return x->shape() == y->shape() ? x : NativeFunc::BroadcastTo(x, Shape(y));
}

NodePtr FuncBuilder::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) {
  return NativeFunc::MatMul(a, b, Value(transpose_a), Value(transpose_b));
}

NodePtr FuncBuilder::MatMulExt(const NodePtr &a, const NodePtr &b) {
  auto [input, mat] = UnifyDtype(a, b);
  return NativeFunc::MatMulExt(input, mat);
}

NodePtr FuncBuilder::Add(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  return NativeFunc::Add(input, other);
}
NodePtr FuncBuilder::Sub(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  return NativeFunc::Sub(input, other);
}
NodePtr FuncBuilder::Mul(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  return NativeFunc::Mul(input, other);
}
NodePtr FuncBuilder::Div(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  return NativeFunc::Div(input, other);
}

NodePtr FuncBuilder::Pow(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, exponent] = UnifyDtype(lhs, rhs);
  return NativeFunc::Pow(input, exponent);
}

NodePtr FuncBuilder::Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto abs = lhs->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    auto [input, other] = UnifyDtype(lhs, rhs);
    auto node = NativeFunc::Equal(input, other);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    return ScalarEq(lhs, rhs, dst_type);
  }
  MS_LOG(EXCEPTION) << "'Equal' only support [Tensor] or [Scalar] input, but got: " << abs->ToString();
}

NodePtr FuncBuilder::NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  auto node = NativeFunc::NotEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  auto node = NativeFunc::GreaterEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  auto node = NativeFunc::Greater(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::LessEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  auto node = NativeFunc::LessEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Less(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype(lhs, rhs);
  auto node = NativeFunc::Less(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Concat(const NodePtr &tensors, const NodePtr &axis) {
  tensors->SetValue(FillZeros(tensors->Value(), tensors->abstract()));
  return NativeFunc::Concat(tensors, axis);
}

NodePtr FuncBuilder::InplaceCopy(const NodePtr &variable, const NodePtr &value, bool non_blocking) {
  return NativeFunc::InplaceCopy(variable, value, Value<bool>(non_blocking));
}

NodePtr FuncBuilder::StackExt(const NodePtr &tensors, const NodePtr &dim) {
  tensors->SetValue(FillZeros(tensors->Value(), tensors->abstract()));
  return NativeFunc::StackExt(tensors, dim);
}

NodePtr FuncBuilder::Tile(const NodePtr &input, const NodePtr &dims) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(dims);
  ValuePtr dims_ptr = dims->BuildValue();
  if (dims_ptr->isa<ValueSequence>()) {
    const auto &dims_seq = dims_ptr->cast<ValueSequencePtr>();
    if (input->shape().size() == 0 && dims_seq->size() == 0) {
      return input;
    }
  }
  return NativeFunc::Tile(input, dims);
}

NodePtr FuncBuilder::InnerCommAllReduce(const NodePtr &grad, const NodePtr &op, const NodePtr &group) {
  MS_LOG(DEBUG) << "Enter InnerCommAllReduce Grad.";
  auto node_ptr = NativeFunc::InnerCommAllReduce(grad, op, group);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
}

NodePtr FuncBuilder::InnerCommAllGather(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group) {
  MS_LOG(DEBUG) << "Enter InnerCommAllGather Grad.";
  auto node_ptr = NativeFunc::InnerCommAllGather(grad, rank_size, group);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
}

NodePtr FuncBuilder::InnerCommReduceScatter(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &type,
                                            const NodePtr &group) {
  MS_LOG(DEBUG) << "Enter InnerCommReduceScatter Grad.";
  auto node_ptr = NativeFunc::InnerCommReduceScatter(grad, rank_size, type, group);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
}

NodePtr FuncBuilder::InnerCommIsend(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group,
                                    const NodePtr &tag) {
  MS_LOG(DEBUG) << "Enter InnerCommIsend Grad.";
  auto node_ptr = NativeFunc::InnerCommIsend(grad, rank_size, group, tag);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
}

NodePtr FuncBuilder::InnerCommIrecv(const NodePtr &tag, const NodePtr &rank_size, const NodePtr &shape,
                                    const NodePtr &group, const NodePtr &type) {
  MS_LOG(DEBUG) << "Enter InnerCommIrecv Grad.";
  auto node_ptr = NativeFunc::InnerCommIrecv(tag, rank_size, shape, group, type);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
}

NodePtr FuncBuilder::InnerCommAllToAllV(const NodePtr &grad, const NodePtr &group, const NodePtr &send_numel_list,
                                        const NodePtr &recv_numel_list, const NodePtr &rank_size,
                                        const NodePtr &split_sizes_empty) {
  MS_LOG(DEBUG) << "Enter InnerCommAllToAllV Grad.";
  auto node_ptr =
    NativeFunc::InnerCommAllToAllV(grad, group, send_numel_list, recv_numel_list, rank_size, split_sizes_empty);
  std::shared_ptr<CommFuncNode> comm_func_ptr = std::dynamic_pointer_cast<CommFuncNode>(node_ptr);
  MS_EXCEPTION_IF_NULL(comm_func_ptr);
  auto handle = comm_func_ptr->comm_handle();
  MS_EXCEPTION_IF_NULL(handle);
  handle->Wait();
  return node_ptr;
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
  MS_EXCEPTION_IF_NULL(node);
  if (!node->Value()->isa<ValueSequence>()) {
    return NewFuncNode(kNone, node->abstract(), InputType::kConstant);
  }
  auto val_seq = node->Value()->cast<ValueSequencePtr>();
  if (val_seq->size() == kSizeZero) {
    return NewFuncNode(kNone, nullptr, InputType::kConstant);
  }
  const auto &value = val_seq->value()[kIndexZero];
  if (!value->isa<tensor::Tensor>()) {
    return NewFuncNode(kNone, nullptr, InputType::kConstant);
  }
  ValuePtrList values(val_seq->size(), kNone);
  return NewFuncNode(std::make_shared<ValueTuple>(values), node->abstract(), InputType::kConstant);
}

ValuePtr FuncBuilder::Ones(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->data_type() != kNumberTypeComplex64 && tensor->data_type() != kNumberTypeComplex128) {
    return Ones(Value<ShapeVector>(tensor->shape()), Value(static_cast<int64_t>(tensor->data_type())))->Value();
  }
  auto ones_abs = CommonUtils::SetAbstractValueToAnyValue(tensor->ToAbstract());
  NodePtr input = NewFuncNode(tensor, ones_abs, InputType::kOpOutput);
  return EmitOp(prim::kPrimOnesLike, {input})->Value();
}

ValuePtr FuncBuilder::Zeros(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->data_type() != kNumberTypeComplex64 && tensor->data_type() != kNumberTypeComplex128) {
    return Zeros(Value<std::vector<int64_t>>(tensor->shape()), Value(static_cast<int64_t>(tensor->data_type())))
      ->Value();
  }
  auto zeros_abs = CommonUtils::SetAbstractValueToAnyValue(tensor->ToAbstract());
  auto input = NewFuncNode(tensor, zeros_abs, InputType::kOpOutput);
  return ZerosLike(input)->Value();
}

ValuePtr FuncBuilder::Add(const ValuePtr &input, const ValuePtr &other) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(other);
  auto input_abs = CommonUtils::SetAbstractValueToAnyValue(input->ToAbstract());
  auto other_abs = CommonUtils::SetAbstractValueToAnyValue(other->ToAbstract());
  auto input_node = NewFuncNode(input, input_abs, InputType::kOpOutput);
  auto other_node = NewFuncNode(other, other_abs, InputType::kOpOutput);
  return NativeFunc::Add(input_node, other_node)->Value();
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
    result = ScalarEqual(lhs->Value(), rhs->Value());
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

void FuncBuilder::ResetInputs() {
  inputs_ptr_ = nullptr;
  attrs_ptr_ = nullptr;
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
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(abs);
  auto convert_value = value;
  if (value->isa<None>()) {
    if (abs->isa<abstract::AbstractTensor>()) {
      auto tensor_dtype = abs->BuildType()->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_dtype);
      auto dtype = tensor_dtype->element();
      auto shape = PyNativeAlgo::Common::BuildShape(abs);
      auto zero_node = Zeros(Value(shape), Value(static_cast<int64_t>(dtype->type_id())));
      convert_value = zero_node->Value();
    } else {
      MS_LOG(DEBUG) << "None value abstract got None abstract!";
    }
  } else if (value->isa<ValueSequence>() && abs->isa<abstract::AbstractSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    auto abs_list = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> value_list;
    value_list.reserve(seq->value().size());
    for (size_t i = 0; i < seq->value().size(); ++i) {
      const auto &val = seq->value()[i];
      const auto &temp_abs = abs_list->elements()[i];
      auto convert = FillZeros(val, temp_abs);
      (void)value_list.emplace_back(convert);
    }
    convert_value = std::make_shared<ValueTuple>(value_list);
  } else if (abs->isa<abstract::AbstractDictionary>()) {
    auto seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq);
    auto abs_list = abs->cast<abstract::AbstractDictionaryPtr>();
    MS_EXCEPTION_IF_NULL(abs_list);
    std::vector<ValuePtr> val_list;
    val_list.reserve(seq->value().size());
    MS_EXCEPTION_IF_CHECK_FAIL(seq->value().size() == abs_list->elements().size(),
                               "Value size should be same as abs size!");
    for (size_t i = 0; i < seq->value().size(); ++i) {
      const auto &val = seq->value()[i];
      const auto &temp_abs = abs_list->elements()[i].second;
      auto convert = FillZeros(val, temp_abs);
      (void)val_list.emplace_back(convert);
    }
    convert_value = std::make_shared<ValueTuple>(val_list);
  }
  return convert_value;
}

NodePtr FuncBuilder::FloorDiv(const NodePtr &input, const NodePtr &other) {
  auto [lhs, rhs] = UnifyDtype(input, other);
  return NativeFunc::FloorDiv(lhs, rhs);
}

// Auto generate
NodePtr FuncBuilder::Ones(const NodePtr &shape, const NodePtr &dtype) { return NativeFunc::Ones(shape, dtype); }

NodePtr FuncBuilder::LerpScalar(const NodePtr &input, const NodePtr &end, const NodePtr &weight) {
  return NativeFunc::LerpScalar(input, end, weight);
}

NodePtr FuncBuilder::Atanh(const NodePtr &input) { return NativeFunc::Atanh(input); }

NodePtr FuncBuilder::ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::ClampScalar(input, min, max);
}

NodePtr FuncBuilder::InplaceRandom(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                                   const NodePtr &offset) {
  return NativeFunc::InplaceRandom(input, from_, to, seed, offset);
}

NodePtr FuncBuilder::ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::ClampTensor(input, min, max);
}

NodePtr FuncBuilder::Kthvalue(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::Kthvalue(input, k, dim, keepdim);
}

NodePtr FuncBuilder::CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
  return NativeFunc::CumsumExt(input, dim, dtype);
}

NodePtr FuncBuilder::SplitTensor(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
  return NativeFunc::SplitTensor(input, split_size, dim);
}

NodePtr FuncBuilder::InplaceUniform(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                                    const NodePtr &offset) {
  return NativeFunc::InplaceUniform(input, from_, to, seed, offset);
}

NodePtr FuncBuilder::RotaryPositionEmbeddingGrad(const NodePtr &dy, const NodePtr &cos, const NodePtr &sin,
                                                 const NodePtr &dx, const NodePtr &mode) {
  return NativeFunc::RotaryPositionEmbeddingGrad(dy, cos, sin, dx, mode);
}

NodePtr FuncBuilder::KLDiv(const NodePtr &input, const NodePtr &target, const NodePtr &reduction,
                           const NodePtr &log_target) {
  return NativeFunc::KLDiv(input, target, reduction, log_target);
}

NodePtr FuncBuilder::OnesLikeExt(const NodePtr &input, const NodePtr &dtype) {
  return NativeFunc::OnesLikeExt(input, dtype);
}

NodePtr FuncBuilder::Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx,
                               const NodePtr &max_norm, const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) {
  return NativeFunc::Embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq);
}

NodePtr FuncBuilder::SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) {
  return NativeFunc::SoftplusExt(input, beta, threshold);
}

NodePtr FuncBuilder::ViewAs(const NodePtr &input, const NodePtr &other) { return NativeFunc::ViewAs(input, other); }

NodePtr FuncBuilder::Cosh(const NodePtr &input) { return NativeFunc::Cosh(input); }

NodePtr FuncBuilder::GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight,
                               const NodePtr &bias, const NodePtr &eps) {
  return NativeFunc::GroupNorm(input, num_groups, weight, bias, eps);
}

NodePtr FuncBuilder::InnerIndex(const NodePtr &input, const NodePtr &indices) {
  return NativeFunc::InnerIndex(input, indices);
}

NodePtr FuncBuilder::InplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                                     const NodePtr &accumulate) {
  return NativeFunc::InplaceIndexPut(input, indices, values, accumulate);
}

NodePtr FuncBuilder::AddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &epsilon) {
  return NativeFunc::AddRmsNorm(x1, x2, gamma, epsilon);
}

NodePtr FuncBuilder::ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad3DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::FlashAttentionScoreGrad(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy, const NodePtr &pse_shift,
  const NodePtr &drop_mask, const NodePtr &padding_mask, const NodePtr &atten_mask, const NodePtr &softmax_max,
  const NodePtr &softmax_sum, const NodePtr &softmax_in, const NodePtr &attention_in, const NodePtr &prefix,
  const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen, const NodePtr &head_num, const NodePtr &keep_prob,
  const NodePtr &scale_value, const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
  const NodePtr &input_layout, const NodePtr &sparse_mode) {
  return NativeFunc::FlashAttentionScoreGrad(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask,
                                             softmax_max, softmax_sum, softmax_in, attention_in, prefix,
                                             actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value,
                                             pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode);
}

NodePtr FuncBuilder::BitwiseNot(const NodePtr &input) { return NativeFunc::BitwiseNot(input); }

NodePtr FuncBuilder::ConvolutionStr(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                    const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                    const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups) {
  return NativeFunc::ConvolutionStr(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

NodePtr FuncBuilder::LogSoftmax(const NodePtr &logits, const NodePtr &axis) {
  return NativeFunc::LogSoftmax(logits, axis);
}

NodePtr FuncBuilder::RemainderScalarTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::RemainderScalarTensor(input, other);
}

NodePtr FuncBuilder::Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                           const NodePtr &alpha) {
  return NativeFunc::Addmm(input, mat1, mat2, beta, alpha);
}

NodePtr FuncBuilder::GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                       const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                       const NodePtr &align_corners, const NodePtr &output_mask) {
  return NativeFunc::GridSampler3DGrad(grad, input_x, grid, interpolation_mode, padding_mode, align_corners,
                                       output_mask);
}

NodePtr FuncBuilder::MoeDistributeDispatch(
  const NodePtr &x, const NodePtr &expert_ids, const NodePtr &ep_world_size, const NodePtr &ep_rank_id,
  const NodePtr &moe_expert_num, const NodePtr &expert_scales, const NodePtr &scales, const NodePtr &x_active_mask,
  const NodePtr &group_ep, const NodePtr &group_tp, const NodePtr &tp_world_size, const NodePtr &tp_rank_id,
  const NodePtr &expert_shard_type, const NodePtr &shared_expert_num, const NodePtr &shared_expert_rank_num,
  const NodePtr &quant_mode, const NodePtr &global_bs, const NodePtr &expert_token_nums_type) {
  return NativeFunc::MoeDistributeDispatch(x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_scales,
                                           scales, x_active_mask, group_ep, group_tp, tp_world_size, tp_rank_id,
                                           expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode,
                                           global_bs, expert_token_nums_type);
}

NodePtr FuncBuilder::Polar(const NodePtr &abs, const NodePtr &angle) { return NativeFunc::Polar(abs, angle); }

NodePtr FuncBuilder::Sqrt(const NodePtr &x) { return NativeFunc::Sqrt(x); }

NodePtr FuncBuilder::TraceExt(const NodePtr &input) { return NativeFunc::TraceExt(input); }

NodePtr FuncBuilder::Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                             const NodePtr &return_counts) {
  return NativeFunc::Unique2(input, sorted, return_inverse, return_counts);
}

NodePtr FuncBuilder::LogSigmoidGrad(const NodePtr &dy, const NodePtr &input, const NodePtr &buffer) {
  return NativeFunc::LogSigmoidGrad(dy, input, buffer);
}

NodePtr FuncBuilder::BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) {
  return NativeFunc::BatchMatMulExt(input, mat2);
}

NodePtr FuncBuilder::RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) {
  return NativeFunc::RepeatInterleaveGrad(input, repeats, dim);
}

NodePtr FuncBuilder::MoeTokenUnpermuteGrad(const NodePtr &permuted_tokens, const NodePtr &unpermuted_tokens_grad,
                                           const NodePtr &sorted_indices, const NodePtr &probs,
                                           const NodePtr &padded_mode, const NodePtr &restore_shape) {
  return NativeFunc::MoeTokenUnpermuteGrad(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, padded_mode,
                                           restore_shape);
}

NodePtr FuncBuilder::FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
  return NativeFunc::FillTensor(size, fill_value, dtype);
}

NodePtr FuncBuilder::AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                   const NodePtr &count_include_pad, const NodePtr &divisor_override) {
  return NativeFunc::AvgPool2DGrad(grad, image, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                   divisor_override);
}

NodePtr FuncBuilder::BitwiseXorTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseXorTensor(input, other);
}

NodePtr FuncBuilder::ReplicationPad2D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad2D(input, padding);
}

NodePtr FuncBuilder::SigmoidGrad(const NodePtr &y, const NodePtr &dy) { return NativeFunc::SigmoidGrad(y, dy); }

NodePtr FuncBuilder::AvgPool3DGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &kernel_size,
                                      const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                      const NodePtr &count_include_pad, const NodePtr &divisor_override) {
  return NativeFunc::AvgPool3DGradExt(grad, input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                      divisor_override);
}

NodePtr FuncBuilder::RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                             const NodePtr &device) {
  return NativeFunc::RandExt(shape, seed, offset, dtype, device);
}

NodePtr FuncBuilder::GreaterEqualScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::GreaterEqualScalar(input, other);
}

NodePtr FuncBuilder::HSigmoidGrad(const NodePtr &grads, const NodePtr &input_x) {
  return NativeFunc::HSigmoidGrad(grads, input_x);
}

NodePtr FuncBuilder::Swiglu(const NodePtr &input, const NodePtr &dim) { return NativeFunc::Swiglu(input, dim); }

NodePtr FuncBuilder::SplitWithSizeView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
  return NativeFunc::SplitWithSizeView(input, split_size, dim);
}

NodePtr FuncBuilder::Squeeze(const NodePtr &input, const NodePtr &axis) { return NativeFunc::Squeeze(input, axis); }

NodePtr FuncBuilder::UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest2D(x, output_size, scales);
}

NodePtr FuncBuilder::Sin(const NodePtr &input) { return NativeFunc::Sin(input); }

NodePtr FuncBuilder::TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                             const NodePtr &sorted) {
  return NativeFunc::TopkExt(input, k, dim, largest, sorted);
}

NodePtr FuncBuilder::BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                            const NodePtr &weight, const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropyGrad(input, target, grad_output, weight, reduction);
}

NodePtr FuncBuilder::SwigluGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &dim) {
  return NativeFunc::SwigluGrad(grad_output, input, dim);
}

NodePtr FuncBuilder::InplaceScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                         const NodePtr &value) {
  return NativeFunc::InplaceScatterValue(input, dim, index, value);
}

NodePtr FuncBuilder::InplaceReLU(const NodePtr &input) { return NativeFunc::InplaceReLU(input); }

NodePtr FuncBuilder::SiLU(const NodePtr &input) { return NativeFunc::SiLU(input); }

NodePtr FuncBuilder::AddLayerNormGrad(const NodePtr &dy, const NodePtr &x1, const NodePtr &x2, const NodePtr &rstd,
                                      const NodePtr &mean, const NodePtr &gamma, const NodePtr &dsumOptional) {
  return NativeFunc::AddLayerNormGrad(dy, x1, x2, rstd, mean, gamma, dsumOptional);
}

NodePtr FuncBuilder::HShrink(const NodePtr &input, const NodePtr &lambd) { return NativeFunc::HShrink(input, lambd); }

NodePtr FuncBuilder::Take(const NodePtr &input, const NodePtr &index) { return NativeFunc::Take(input, index); }

NodePtr FuncBuilder::Std(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
  return NativeFunc::Std(input, dim, correction, keepdim);
}

NodePtr FuncBuilder::InplaceErfinv(const NodePtr &input) { return NativeFunc::InplaceErfinv(input); }

NodePtr FuncBuilder::ToDevice(const NodePtr &input, const NodePtr &device, const NodePtr &dtype,
                              const NodePtr &non_blocking, const NodePtr &copy) {
  return NativeFunc::ToDevice(input, device, dtype, non_blocking, copy);
}

NodePtr FuncBuilder::FmodTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::FmodTensor(input, other);
}

NodePtr FuncBuilder::MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) {
  return NativeFunc::MaskedFill(input_x, mask, value);
}

NodePtr FuncBuilder::InplaceTanh(const NodePtr &input) { return NativeFunc::InplaceTanh(input); }

NodePtr FuncBuilder::Expm1(const NodePtr &input) { return NativeFunc::Expm1(input); }

NodePtr FuncBuilder::InplaceMaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) {
  return NativeFunc::InplaceMaskedScatter(input, mask, source);
}

NodePtr FuncBuilder::Neg(const NodePtr &input) { return NativeFunc::Neg(input); }

NodePtr FuncBuilder::InplaceBernoulliTensor(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                            const NodePtr &offset) {
  return NativeFunc::InplaceBernoulliTensor(input, p, seed, offset);
}

NodePtr FuncBuilder::DiagonalView(const NodePtr &input, const NodePtr &offset, const NodePtr &dim1,
                                  const NodePtr &dim2) {
  return NativeFunc::DiagonalView(input, offset, dim1, dim2);
}

NodePtr FuncBuilder::FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
  return NativeFunc::FillScalar(size, fill_value, dtype);
}

NodePtr FuncBuilder::AdaptiveMaxPool1D(const NodePtr &input, const NodePtr &output_size) {
  return NativeFunc::AdaptiveMaxPool1D(input, output_size);
}

NodePtr FuncBuilder::LinalgQr(const NodePtr &A, const NodePtr &mode) { return NativeFunc::LinalgQr(A, mode); }

NodePtr FuncBuilder::ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ArgMinWithValue(input, axis, keep_dims);
}

NodePtr FuncBuilder::L1LossBackwardExt(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                                       const NodePtr &reduction) {
  return NativeFunc::L1LossBackwardExt(grad_output, input, target, reduction);
}

NodePtr FuncBuilder::ReflectionPad2D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad2D(input, padding);
}

NodePtr FuncBuilder::LogicalXor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::LogicalXor(input, other);
}

NodePtr FuncBuilder::Cummax(const NodePtr &input, const NodePtr &axis) { return NativeFunc::Cummax(input, axis); }

NodePtr FuncBuilder::Minimum(const NodePtr &input, const NodePtr &other) { return NativeFunc::Minimum(input, other); }

NodePtr FuncBuilder::AdaptiveAvgPool2DExt(const NodePtr &input, const NodePtr &output_size) {
  return NativeFunc::AdaptiveAvgPool2DExt(input, output_size);
}

NodePtr FuncBuilder::GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) {
  return NativeFunc::GatherDGradV2(x, dim, index, dout);
}

NodePtr FuncBuilder::SmoothL1Loss(const NodePtr &prediction, const NodePtr &target, const NodePtr &beta,
                                  const NodePtr &reduction) {
  return NativeFunc::SmoothL1Loss(prediction, target, beta, reduction);
}

NodePtr FuncBuilder::CumminExt(const NodePtr &input, const NodePtr &dim) { return NativeFunc::CumminExt(input, dim); }

NodePtr FuncBuilder::BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                       const NodePtr &posWeight, const NodePtr &reduction) {
  return NativeFunc::BCEWithLogitsLoss(input, target, weight, posWeight, reduction);
}

NodePtr FuncBuilder::BroadcastToView(const NodePtr &input, const NodePtr &shape) {
  return NativeFunc::BroadcastToView(input, shape);
}

NodePtr FuncBuilder::RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset,
                                 const NodePtr &dtype, const NodePtr &device) {
  return NativeFunc::RandLikeExt(tensor, seed, offset, dtype, device);
}

NodePtr FuncBuilder::InplaceExp(const NodePtr &input) { return NativeFunc::InplaceExp(input); }

NodePtr FuncBuilder::BitwiseAndTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseAndTensor(input, other);
}

NodePtr FuncBuilder::UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest3DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::MultiScaleDeformableAttn(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                              const NodePtr &locations, const NodePtr &weight) {
  return NativeFunc::MultiScaleDeformableAttn(value, shape, offset, locations, weight);
}

NodePtr FuncBuilder::LogicalOr(const NodePtr &x, const NodePtr &y) { return NativeFunc::LogicalOr(x, y); }

NodePtr FuncBuilder::MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                     const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                     const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolWithMask(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::InplaceFloorDivides(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceFloorDivides(input, other);
}

NodePtr FuncBuilder::ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) {
  return NativeFunc::ScatterAddExt(input, dim, index, src);
}

NodePtr FuncBuilder::ReflectionPad3D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad3D(input, padding);
}

NodePtr FuncBuilder::HSwishGrad(const NodePtr &y_grad, const NodePtr &x) { return NativeFunc::HSwishGrad(y_grad, x); }

NodePtr FuncBuilder::FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) {
  return NativeFunc::FlattenExt(input, start_dim, end_dim);
}

NodePtr FuncBuilder::Square(const NodePtr &input) { return NativeFunc::Square(input); }

NodePtr FuncBuilder::Addbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                            const NodePtr &alpha) {
  return NativeFunc::Addbmm(input, batch1, batch2, beta, alpha);
}

NodePtr FuncBuilder::Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) {
  return NativeFunc::Arange(start, end, step, dtype);
}

NodePtr FuncBuilder::InplaceIndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                            const NodePtr &value) {
  return NativeFunc::InplaceIndexFillTensor(input, dim, index, value);
}

NodePtr FuncBuilder::Round(const NodePtr &input, const NodePtr &decimals) { return NativeFunc::Round(input, decimals); }

NodePtr FuncBuilder::SliceExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                                  const NodePtr &step) {
  return NativeFunc::SliceExtView(input, dim, start, end, step);
}

NodePtr FuncBuilder::ArgMinExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::ArgMinExt(input, dim, keepdim);
}

NodePtr FuncBuilder::ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad1DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::MaskedSelectGrad(const NodePtr &input, const NodePtr &mask, const NodePtr &grad) {
  return NativeFunc::MaskedSelectGrad(input, mask, grad);
}

NodePtr FuncBuilder::SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::SubExt(input, other, alpha);
}

NodePtr FuncBuilder::InnerMoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices,
                                            const NodePtr &probs, const NodePtr &padded_mode,
                                            const NodePtr &restore_shape) {
  return NativeFunc::InnerMoeTokenUnpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

NodePtr FuncBuilder::SelectExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
  return NativeFunc::SelectExtView(input, dim, index);
}

NodePtr FuncBuilder::InplaceMaskedFillTensor(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
  return NativeFunc::InplaceMaskedFillTensor(input, mask, value);
}

NodePtr FuncBuilder::InplaceDivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
  return NativeFunc::InplaceDivMod(input, other, rounding_mode);
}

NodePtr FuncBuilder::NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                       const NodePtr &offset) {
  return NativeFunc::NormalFloatTensor(mean, std, seed, offset);
}

NodePtr FuncBuilder::SplitTensorView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
  return NativeFunc::SplitTensorView(input, split_size, dim);
}

NodePtr FuncBuilder::ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad2DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::Sign(const NodePtr &input) { return NativeFunc::Sign(input); }

NodePtr FuncBuilder::Narrow(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) {
  return NativeFunc::Narrow(input, dim, start, length);
}

NodePtr FuncBuilder::GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                   const NodePtr &padding_mode, const NodePtr &align_corners) {
  return NativeFunc::GridSampler3D(input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                                    const NodePtr &epsilon, const NodePtr &additionalOut) {
  return NativeFunc::AddLayerNormV2(x1, x2, gamma, beta, epsilon, additionalOut);
}

NodePtr FuncBuilder::IsInf(const NodePtr &input) { return NativeFunc::IsInf(input); }

NodePtr FuncBuilder::InplaceIndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                        const NodePtr &source, const NodePtr &alpha) {
  return NativeFunc::InplaceIndexAddExt(input, dim, index, source, alpha);
}

NodePtr FuncBuilder::BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                      const NodePtr &running_mean, const NodePtr &running_var,
                                      const NodePtr &saved_mean, const NodePtr &saved_rstd, const NodePtr &training,
                                      const NodePtr &eps, const NodePtr &output_mask) {
  return NativeFunc::BatchNormGradExt(dout, input, weight, running_mean, running_var, saved_mean, saved_rstd, training,
                                      eps, output_mask);
}

NodePtr FuncBuilder::DivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
  return NativeFunc::DivMod(input, other, rounding_mode);
}

NodePtr FuncBuilder::Slice(const NodePtr &input, const NodePtr &begin, const NodePtr &size) {
  return NativeFunc::Slice(input, begin, size);
}

NodePtr FuncBuilder::RandIntLike(const NodePtr &input, const NodePtr &low, const NodePtr &high, const NodePtr &seed,
                                 const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) {
  return NativeFunc::RandIntLike(input, low, high, seed, offset, dtype, device);
}

NodePtr FuncBuilder::AsStrided(const NodePtr &input, const NodePtr &size, const NodePtr &stride,
                               const NodePtr &storage_offset) {
  return NativeFunc::AsStrided(input, size, stride, storage_offset);
}

NodePtr FuncBuilder::MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                            const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                            const NodePtr &dilation, const NodePtr &ceil_mode,
                                            const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolGradWithIndices(x, grad, argmax, kernel_size, strides, pads, dilation, ceil_mode,
                                            argmax_type);
}

NodePtr FuncBuilder::RemainderTensorScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::RemainderTensorScalar(input, other);
}

NodePtr FuncBuilder::FmodScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::FmodScalar(input, other);
}

NodePtr FuncBuilder::Randn(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                           const NodePtr &device) {
  return NativeFunc::Randn(shape, seed, offset, dtype, device);
}

NodePtr FuncBuilder::BitwiseXorScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseXorScalar(input, other);
}

NodePtr FuncBuilder::UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                         const NodePtr &align_corners) {
  return NativeFunc::UpsampleTrilinear3D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ArgMaxWithValue(input, axis, keep_dims);
}

NodePtr FuncBuilder::InplaceFloor(const NodePtr &input) { return NativeFunc::InplaceFloor(input); }

NodePtr FuncBuilder::UnstackExtView(const NodePtr &input, const NodePtr &dim) {
  return NativeFunc::UnstackExtView(input, dim);
}

NodePtr FuncBuilder::InplaceFloorDivide(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceFloorDivide(input, other);
}

NodePtr FuncBuilder::InplaceSubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::InplaceSubExt(input, other, alpha);
}

NodePtr FuncBuilder::GeLU(const NodePtr &input) { return NativeFunc::GeLU(input); }

NodePtr FuncBuilder::ReplicationPad1D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad1D(input, padding);
}

NodePtr FuncBuilder::InplaceCopy(const NodePtr &input, const NodePtr &src, const NodePtr &non_blocking) {
  return NativeFunc::InplaceCopy(input, src, non_blocking);
}

NodePtr FuncBuilder::Baddbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                             const NodePtr &alpha) {
  return NativeFunc::Baddbmm(input, batch1, batch2, beta, alpha);
}

NodePtr FuncBuilder::ExpandDims(const NodePtr &input_x, const NodePtr &axis) {
  return NativeFunc::ExpandDims(input_x, axis);
}

NodePtr FuncBuilder::LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) {
  return NativeFunc::LeakyReLUExt(input, negative_slope);
}

NodePtr FuncBuilder::UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                               const NodePtr &dim) {
  return NativeFunc::UniqueDim(input, sorted, return_inverse, dim);
}

NodePtr FuncBuilder::HistcExt(const NodePtr &input, const NodePtr &bins, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::HistcExt(input, bins, min, max);
}

NodePtr FuncBuilder::IncreFlashAttention(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &attn_mask,
  const NodePtr &actual_seq_lengths, const NodePtr &pse_shift, const NodePtr &dequant_scale1,
  const NodePtr &quant_scale1, const NodePtr &dequant_scale2, const NodePtr &quant_scale2, const NodePtr &quant_offset2,
  const NodePtr &antiquant_scale, const NodePtr &antiquant_offset, const NodePtr &block_table,
  const NodePtr &kv_padding_size, const NodePtr &num_heads, const NodePtr &input_layout, const NodePtr &scale_value,
  const NodePtr &num_key_value_heads, const NodePtr &block_size, const NodePtr &inner_precise) {
  return NativeFunc::IncreFlashAttention(query, key, value, attn_mask, actual_seq_lengths, pse_shift, dequant_scale1,
                                         quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale,
                                         antiquant_offset, block_table, kv_padding_size, num_heads, input_layout,
                                         scale_value, num_key_value_heads, block_size, inner_precise);
}

NodePtr FuncBuilder::Log10(const NodePtr &input) { return NativeFunc::Log10(input); }

NodePtr FuncBuilder::EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                            const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) {
  return NativeFunc::EmbeddingDenseBackward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

NodePtr FuncBuilder::Mm(const NodePtr &input, const NodePtr &mat2) { return NativeFunc::Mm(input, mat2); }

NodePtr FuncBuilder::Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                               const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Col2ImExt(input, output_size, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) {
  return NativeFunc::GeLUGrad(dy, x, y);
}

NodePtr FuncBuilder::OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                               const NodePtr &off_value, const NodePtr &axis) {
  return NativeFunc::OneHotExt(tensor, num_classes, on_value, off_value, axis);
}

NodePtr FuncBuilder::SiLUGrad(const NodePtr &dout, const NodePtr &x) { return NativeFunc::SiLUGrad(dout, x); }

NodePtr FuncBuilder::ConvolutionStrGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                        const NodePtr &bias, const NodePtr &stride, const NodePtr &padding,
                                        const NodePtr &dilation, const NodePtr &transposed,
                                        const NodePtr &output_padding, const NodePtr &groups,
                                        const NodePtr &output_mask) {
  return NativeFunc::ConvolutionStrGrad(dout, input, weight, bias, stride, padding, dilation, transposed,
                                        output_padding, groups, output_mask);
}

NodePtr FuncBuilder::InplaceDivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
  return NativeFunc::InplaceDivMods(input, other, rounding_mode);
}

NodePtr FuncBuilder::SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending,
                             const NodePtr &stable) {
  return NativeFunc::SortExt(input, dim, descending, stable);
}

NodePtr FuncBuilder::Generator(const NodePtr &cmd, const NodePtr &inputs) { return NativeFunc::Generator(cmd, inputs); }

NodePtr FuncBuilder::LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) {
  return NativeFunc::LinSpaceExt(start, end, steps, dtype);
}

NodePtr FuncBuilder::InnerUnique(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse) {
  return NativeFunc::InnerUnique(input, sorted, return_inverse);
}

NodePtr FuncBuilder::AddcdivExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                                const NodePtr &value) {
  return NativeFunc::AddcdivExt(input, tensor1, tensor2, value);
}

NodePtr FuncBuilder::LogAddExp2(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::LogAddExp2(input, other);
}

NodePtr FuncBuilder::ThresholdGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &threshold) {
  return NativeFunc::ThresholdGrad(grad_output, input, threshold);
}

NodePtr FuncBuilder::LogSoftmaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
  return NativeFunc::LogSoftmaxExt(input, dim, dtype);
}

NodePtr FuncBuilder::PowScalarTensor(const NodePtr &input, const NodePtr &exponent) {
  return NativeFunc::PowScalarTensor(input, exponent);
}

NodePtr FuncBuilder::AvgPool3DExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                                  const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                                  const NodePtr &divisor_override) {
  return NativeFunc::AvgPool3DExt(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

NodePtr FuncBuilder::InplaceFillDiagonal(const NodePtr &input, const NodePtr &fill_value, const NodePtr &wrap) {
  return NativeFunc::InplaceFillDiagonal(input, fill_value, wrap);
}

NodePtr FuncBuilder::Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                                const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Col2ImGrad(input, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::AllGatherMatmul(const NodePtr &input, const NodePtr &x2, const NodePtr &group,
                                     const NodePtr &world_size, const NodePtr &bias, const NodePtr &gather_index,
                                     const NodePtr &gather_output, const NodePtr &comm_turn, const NodePtr &trans_input,
                                     const NodePtr &trans_x2) {
  return NativeFunc::AllGatherMatmul(input, x2, group, world_size, bias, gather_index, gather_output, comm_turn,
                                     trans_input, trans_x2);
}

NodePtr FuncBuilder::MaxUnpool2DExt(const NodePtr &input, const NodePtr &indices, const NodePtr &kernel_size,
                                    const NodePtr &stride, const NodePtr &padding, const NodePtr &output_size) {
  return NativeFunc::MaxUnpool2DExt(input, indices, kernel_size, stride, padding, output_size);
}

NodePtr FuncBuilder::InplaceGroupedMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                                             const NodePtr &out) {
  return NativeFunc::InplaceGroupedMatmulAdd(x, weight, group_list, out);
}

NodePtr FuncBuilder::MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                        const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                        const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolWithIndices(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) {
  return NativeFunc::SoftmaxBackward(dout, out, dim);
}

NodePtr FuncBuilder::MatrixInverseExt(const NodePtr &input) { return NativeFunc::MatrixInverseExt(input); }

NodePtr FuncBuilder::Tanh(const NodePtr &input) { return NativeFunc::Tanh(input); }

NodePtr FuncBuilder::DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
  return NativeFunc::DropoutGradExt(input, mask, p);
}

NodePtr FuncBuilder::InnerNonZero(const NodePtr &input) { return NativeFunc::InnerNonZero(input); }

NodePtr FuncBuilder::AllFinite(const NodePtr &tensors) { return NativeFunc::AllFinite(tensors); }

NodePtr FuncBuilder::ReshapeAndCache(const NodePtr &key, const NodePtr &value, const NodePtr &key_cache,
                                     const NodePtr &value_cache, const NodePtr &slot_mapping) {
  return NativeFunc::ReshapeAndCache(key, value, key_cache, value_cache, slot_mapping);
}

NodePtr FuncBuilder::InplaceClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::InplaceClampScalar(input, min, max);
}

NodePtr FuncBuilder::NewOnes(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) {
  return NativeFunc::NewOnes(input, size, dtype);
}

NodePtr FuncBuilder::Dot(const NodePtr &input, const NodePtr &other) { return NativeFunc::Dot(input, other); }

NodePtr FuncBuilder::InplaceAddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::InplaceAddExt(input, other, alpha);
}

NodePtr FuncBuilder::XLogYScalarOther(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::XLogYScalarOther(input, other);
}

NodePtr FuncBuilder::AvgPool1D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad) {
  return NativeFunc::AvgPool1D(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

NodePtr FuncBuilder::RotaryPositionEmbedding(const NodePtr &x, const NodePtr &cos, const NodePtr &sin,
                                             const NodePtr &mode) {
  return NativeFunc::RotaryPositionEmbedding(x, cos, sin, mode);
}

NodePtr FuncBuilder::RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) {
  return NativeFunc::RmsNorm(x, gamma, epsilon);
}

NodePtr FuncBuilder::InplaceZero(const NodePtr &input) { return NativeFunc::InplaceZero(input); }

NodePtr FuncBuilder::ExpandDimsView(const NodePtr &input, const NodePtr &dim) {
  return NativeFunc::ExpandDimsView(input, dim);
}

NodePtr FuncBuilder::Outer(const NodePtr &input, const NodePtr &vec2) { return NativeFunc::Outer(input, vec2); }

NodePtr FuncBuilder::InplaceLog(const NodePtr &input) { return NativeFunc::InplaceLog(input); }

NodePtr FuncBuilder::ToOther(const NodePtr &input, const NodePtr &other, const NodePtr &non_blocking,
                             const NodePtr &copy) {
  return NativeFunc::ToOther(input, other, non_blocking, copy);
}

NodePtr FuncBuilder::InplaceAddmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                                  const NodePtr &alpha) {
  return NativeFunc::InplaceAddmm(input, mat1, mat2, beta, alpha);
}

NodePtr FuncBuilder::InplaceThreshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) {
  return NativeFunc::InplaceThreshold(input, threshold, value);
}

NodePtr FuncBuilder::IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                             const NodePtr &equal_nan) {
  return NativeFunc::IsClose(input, other, rtol, atol, equal_nan);
}

NodePtr FuncBuilder::GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                       const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                       const NodePtr &align_corners, const NodePtr &output_mask) {
  return NativeFunc::GridSampler2DGrad(grad, input_x, grid, interpolation_mode, padding_mode, align_corners,
                                       output_mask);
}

NodePtr FuncBuilder::ReflectionPad1D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad1D(input, padding);
}

NodePtr FuncBuilder::InplaceIndexCopy(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                      const NodePtr &tensor) {
  return NativeFunc::InplaceIndexCopy(input, dim, index, tensor);
}

NodePtr FuncBuilder::InplaceStopGradient(const NodePtr &input) { return NativeFunc::InplaceStopGradient(input); }

NodePtr FuncBuilder::BernoulliExt(const NodePtr &input, const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::BernoulliExt(input, seed, offset);
}

NodePtr FuncBuilder::InplaceDiv(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceDiv(input, other);
}

NodePtr FuncBuilder::Log1p(const NodePtr &input) { return NativeFunc::Log1p(input); }

NodePtr FuncBuilder::SubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::SubScalar(input, other, alpha);
}

NodePtr FuncBuilder::Addmv(const NodePtr &input, const NodePtr &mat, const NodePtr &vec, const NodePtr &beta,
                           const NodePtr &alpha) {
  return NativeFunc::Addmv(input, mat, vec, beta, alpha);
}

NodePtr FuncBuilder::SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                                  const NodePtr &dtype, const NodePtr &right) {
  return NativeFunc::SearchSorted(sorted_sequence, values, sorter, dtype, right);
}

NodePtr FuncBuilder::UpsampleBicubic2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                       const NodePtr &align_corners) {
  return NativeFunc::UpsampleBicubic2D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) {
  return NativeFunc::GatherD(x, dim, index);
}

NodePtr FuncBuilder::Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                             const NodePtr &reduce) {
  return NativeFunc::Scatter(input, dim, index, src, reduce);
}

NodePtr FuncBuilder::AcoshExt(const NodePtr &input) { return NativeFunc::AcoshExt(input); }

NodePtr FuncBuilder::Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                 const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                 const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups) {
  return NativeFunc::Convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

NodePtr FuncBuilder::Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
  return NativeFunc::Chunk(input, chunks, dim);
}

NodePtr FuncBuilder::Clone(const NodePtr &input) { return NativeFunc::Clone(input); }

NodePtr FuncBuilder::ReLU(const NodePtr &input) { return NativeFunc::ReLU(input); }

NodePtr FuncBuilder::VarMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction,
                             const NodePtr &keepdim) {
  return NativeFunc::VarMean(input, dim, correction, keepdim);
}

NodePtr FuncBuilder::InplaceFillScalar(const NodePtr &input, const NodePtr &value) {
  return NativeFunc::InplaceFillScalar(input, value);
}

NodePtr FuncBuilder::MultinomialExt(const NodePtr &input, const NodePtr &num_samples, const NodePtr &replacement,
                                    const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::MultinomialExt(input, num_samples, replacement, seed, offset);
}

NodePtr FuncBuilder::MishGradExt(const NodePtr &dout, const NodePtr &x) { return NativeFunc::MishGradExt(dout, x); }

NodePtr FuncBuilder::ReduceMax(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceMax(x, axis, keep_dims);
}

NodePtr FuncBuilder::ArgSort(const NodePtr &input, const NodePtr &dim, const NodePtr &descending,
                             const NodePtr &stable) {
  return NativeFunc::ArgSort(input, dim, descending, stable);
}

NodePtr FuncBuilder::GeluGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &approximate) {
  return NativeFunc::GeluGradExt(grad, input, approximate);
}

NodePtr FuncBuilder::BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input,
                                                          const NodePtr &target, const NodePtr &weight,
                                                          const NodePtr &posWeight, const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropyWithLogitsBackward(grad_output, input, target, weight, posWeight, reduction);
}

NodePtr FuncBuilder::LinalgVectorNorm(const NodePtr &x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
                                      const NodePtr &dtype) {
  return NativeFunc::LinalgVectorNorm(x, ord, dim, keepdim, dtype);
}

NodePtr FuncBuilder::Norm(const NodePtr &input, const NodePtr &p, const NodePtr &dim, const NodePtr &keepdim,
                          const NodePtr &dtype) {
  return NativeFunc::Norm(input, p, dim, keepdim, dtype);
}

NodePtr FuncBuilder::BatchNormElemtGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean,
                                        const NodePtr &invstd, const NodePtr &weight, const NodePtr &sumd_dy,
                                        const NodePtr &sum_dy_xmu, const NodePtr &count) {
  return NativeFunc::BatchNormElemtGrad(dout, input, mean, invstd, weight, sumd_dy, sum_dy_xmu, count);
}

NodePtr FuncBuilder::RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                            const NodePtr &output_size) {
  return NativeFunc::RepeatInterleaveTensor(input, repeats, dim, output_size);
}

NodePtr FuncBuilder::TrilExt(const NodePtr &input, const NodePtr &diagonal) {
  return NativeFunc::TrilExt(input, diagonal);
}

NodePtr FuncBuilder::PReLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &weight) {
  return NativeFunc::PReLUGrad(dy, x, weight);
}

NodePtr FuncBuilder::InplaceScatterSrcReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                             const NodePtr &src, const NodePtr &reduce) {
  return NativeFunc::InplaceScatterSrcReduce(input, dim, index, src, reduce);
}

NodePtr FuncBuilder::AdaptiveAvgPool3DGradExt(const NodePtr &input_grad, const NodePtr &input) {
  return NativeFunc::AdaptiveAvgPool3DGradExt(input_grad, input);
}

NodePtr FuncBuilder::BitwiseOrScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseOrScalar(input, other);
}

NodePtr FuncBuilder::InplaceNormal(const NodePtr &input, const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                   const NodePtr &offset) {
  return NativeFunc::InplaceNormal(input, mean, std, seed, offset);
}

NodePtr FuncBuilder::CountNonZero(const NodePtr &input, const NodePtr &dim) {
  return NativeFunc::CountNonZero(input, dim);
}

NodePtr FuncBuilder::EqualExt(const NodePtr &input, const NodePtr &other) { return NativeFunc::EqualExt(input, other); }

NodePtr FuncBuilder::StdMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction,
                             const NodePtr &keepdim) {
  return NativeFunc::StdMean(input, dim, correction, keepdim);
}

NodePtr FuncBuilder::BatchNormReduceGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean,
                                         const NodePtr &invstd, const NodePtr &weight, const NodePtr &input_g,
                                         const NodePtr &weight_g, const NodePtr &bias_g) {
  return NativeFunc::BatchNormReduceGrad(dout, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

NodePtr FuncBuilder::GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                                   const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                                   const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) {
  return NativeFunc::GroupNormGrad(dy, x, mean, rstd, gamma_opt, num_groups, dx_is_require, dgamma_is_require,
                                   dbeta_is_require);
}

NodePtr FuncBuilder::TanhGrad(const NodePtr &y, const NodePtr &dy) { return NativeFunc::TanhGrad(y, dy); }

NodePtr FuncBuilder::MaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) {
  return NativeFunc::MaskedScatter(input, mask, source);
}

NodePtr FuncBuilder::Exp(const NodePtr &input) { return NativeFunc::Exp(input); }

NodePtr FuncBuilder::BitwiseOrTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseOrTensor(input, other);
}

NodePtr FuncBuilder::NLLLoss2d(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                               const NodePtr &reduction, const NodePtr &ignore_index) {
  return NativeFunc::NLLLoss2d(input, target, weight, reduction, ignore_index);
}

NodePtr FuncBuilder::BatchNormElemt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                    const NodePtr &mean, const NodePtr &invstd, const NodePtr &eps) {
  return NativeFunc::BatchNormElemt(input, weight, bias, mean, invstd, eps);
}

NodePtr FuncBuilder::Hardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) {
  return NativeFunc::Hardtanh(input, min_val, max_val);
}

NodePtr FuncBuilder::Exp2(const NodePtr &input) { return NativeFunc::Exp2(input); }

NodePtr FuncBuilder::Cos(const NodePtr &input) { return NativeFunc::Cos(input); }

NodePtr FuncBuilder::SmoothL1LossGrad(const NodePtr &prediction, const NodePtr &target, const NodePtr &dout,
                                      const NodePtr &beta, const NodePtr &reduction) {
  return NativeFunc::SmoothL1LossGrad(prediction, target, dout, beta, reduction);
}

NodePtr FuncBuilder::MishExt(const NodePtr &input) { return NativeFunc::MishExt(input); }

NodePtr FuncBuilder::Select(const NodePtr &condition, const NodePtr &input, const NodePtr &other) {
  return NativeFunc::Select(condition, input, other);
}

NodePtr FuncBuilder::TransposeView(const NodePtr &input, const NodePtr &input_perm) {
  return NativeFunc::TransposeView(input, input_perm);
}

NodePtr FuncBuilder::AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::AddExt(input, other, alpha);
}

NodePtr FuncBuilder::TransposeExtView(const NodePtr &input, const NodePtr &dim0, const NodePtr &dim1) {
  return NativeFunc::TransposeExtView(input, dim0, dim1);
}

NodePtr FuncBuilder::ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) {
  return NativeFunc::ZerosLikeExt(input, dtype);
}

NodePtr FuncBuilder::NewZeros(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) {
  return NativeFunc::NewZeros(input, size, dtype);
}

NodePtr FuncBuilder::Roll(const NodePtr &input, const NodePtr &shifts, const NodePtr &dims) {
  return NativeFunc::Roll(input, shifts, dims);
}

NodePtr FuncBuilder::InplaceClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::InplaceClampTensor(input, min, max);
}

NodePtr FuncBuilder::ExpandAs(const NodePtr &input, const NodePtr &other) { return NativeFunc::ExpandAs(input, other); }

NodePtr FuncBuilder::Conv1DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
  return NativeFunc::Conv1DExt(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad3DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                               const NodePtr &divisor_override) {
  return NativeFunc::AvgPool2D(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

NodePtr FuncBuilder::FlashAttentionScore(const NodePtr &query, const NodePtr &key, const NodePtr &value,
                                         const NodePtr &real_shift, const NodePtr &drop_mask,
                                         const NodePtr &padding_mask, const NodePtr &attn_mask, const NodePtr &prefix,
                                         const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                                         const NodePtr &head_num, const NodePtr &keep_prob, const NodePtr &scale_value,
                                         const NodePtr &pre_tokens, const NodePtr &next_tokens,
                                         const NodePtr &inner_precise, const NodePtr &input_layout,
                                         const NodePtr &sparse_mode) {
  return NativeFunc::FlashAttentionScore(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix,
                                         actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value,
                                         pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode);
}

NodePtr FuncBuilder::BatchNormGatherStatsWithCounts(const NodePtr &input, const NodePtr &mean, const NodePtr &invstd,
                                                    const NodePtr &running_mean, const NodePtr &running_var,
                                                    const NodePtr &momentum, const NodePtr &eps,
                                                    const NodePtr &counts) {
  return NativeFunc::BatchNormGatherStatsWithCounts(input, mean, invstd, running_mean, running_var, momentum, eps,
                                                    counts);
}

NodePtr FuncBuilder::AtanExt(const NodePtr &input) { return NativeFunc::AtanExt(input); }

NodePtr FuncBuilder::Log2(const NodePtr &input) { return NativeFunc::Log2(input); }

NodePtr FuncBuilder::RandpermExt(const NodePtr &n, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) {
  return NativeFunc::RandpermExt(n, seed, offset, dtype);
}

NodePtr FuncBuilder::LogAddExp(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::LogAddExp(input, other);
}

NodePtr FuncBuilder::LogSigmoid(const NodePtr &input) { return NativeFunc::LogSigmoid(input); }

NodePtr FuncBuilder::XLogYScalarSelf(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::XLogYScalarSelf(input, other);
}

NodePtr FuncBuilder::TriangularSolve(const NodePtr &b, const NodePtr &A, const NodePtr &upper, const NodePtr &transpose,
                                     const NodePtr &unitriangular) {
  return NativeFunc::TriangularSolve(b, A, upper, transpose, unitriangular);
}

NodePtr FuncBuilder::SpeedFusionAttention(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &head_num, const NodePtr &input_layout,
  const NodePtr &seed, const NodePtr &offset, const NodePtr &pse, const NodePtr &padding_mask,
  const NodePtr &atten_mask, const NodePtr &scale, const NodePtr &keep_prob, const NodePtr &pre_tokens,
  const NodePtr &next_tokens, const NodePtr &inner_precise, const NodePtr &prefix, const NodePtr &actual_seq_qlen,
  const NodePtr &actual_seq_kvlen, const NodePtr &sparse_mode, const NodePtr &gen_mask_parallel, const NodePtr &sync,
  const NodePtr &pse_type, const NodePtr &q_start_idx, const NodePtr &kv_start_idx) {
  return NativeFunc::SpeedFusionAttention(query, key, value, head_num, input_layout, seed, offset, pse, padding_mask,
                                          atten_mask, scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix,
                                          actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync,
                                          pse_type, q_start_idx, kv_start_idx);
}

NodePtr FuncBuilder::GluGrad(const NodePtr &grads, const NodePtr &x, const NodePtr &axis) {
  return NativeFunc::GluGrad(grads, x, axis);
}

NodePtr FuncBuilder::IsNegInf(const NodePtr &input) { return NativeFunc::IsNegInf(input); }

NodePtr FuncBuilder::DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed,
                                       const NodePtr &offset, const NodePtr &dtype) {
  return NativeFunc::DropoutGenMaskExt(shape, p, seed, offset, dtype);
}

NodePtr FuncBuilder::HShrinkGrad(const NodePtr &gradients, const NodePtr &features, const NodePtr &lambd) {
  return NativeFunc::HShrinkGrad(gradients, features, lambd);
}

NodePtr FuncBuilder::EmptyLike(const NodePtr &input, const NodePtr &dtype, const NodePtr &device,
                               const NodePtr &pin_memory) {
  return NativeFunc::EmptyLike(input, dtype, device, pin_memory);
}

NodePtr FuncBuilder::MeanExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
  return NativeFunc::MeanExt(input, dim, keepdim, dtype);
}

NodePtr FuncBuilder::InplaceScatterAdd(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                       const NodePtr &src) {
  return NativeFunc::InplaceScatterAdd(input, dim, index, src);
}

NodePtr FuncBuilder::InplaceMul(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceMul(input, other);
}

NodePtr FuncBuilder::LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                                  const NodePtr &bias, const NodePtr &eps) {
  return NativeFunc::LayerNormExt(input, normalized_shape, weight, bias, eps);
}

NodePtr FuncBuilder::LogicalAnd(const NodePtr &x, const NodePtr &y) { return NativeFunc::LogicalAnd(x, y); }

NodePtr FuncBuilder::Divs(const NodePtr &input, const NodePtr &other) { return NativeFunc::Divs(input, other); }

NodePtr FuncBuilder::InnerInplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                                          const NodePtr &accumulate) {
  return NativeFunc::InnerInplaceIndexPut(input, indices, values, accumulate);
}

NodePtr FuncBuilder::InplaceIndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                            const NodePtr &value) {
  return NativeFunc::InplaceIndexFillScalar(input, dim, index, value);
}

NodePtr FuncBuilder::NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                       const NodePtr &offset) {
  return NativeFunc::NormalTensorFloat(mean, std, seed, offset);
}

NodePtr FuncBuilder::AdaptiveAvgPool2DGradExt(const NodePtr &grad_output, const NodePtr &x) {
  return NativeFunc::AdaptiveAvgPool2DGradExt(grad_output, x);
}

NodePtr FuncBuilder::ProdExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
  return NativeFunc::ProdExt(input, dim, keepdim, dtype);
}

NodePtr FuncBuilder::Softmax(const NodePtr &input, const NodePtr &axis) { return NativeFunc::Softmax(input, axis); }

NodePtr FuncBuilder::InplaceElu(const NodePtr &input, const NodePtr &alpha) {
  return NativeFunc::InplaceElu(input, alpha);
}

NodePtr FuncBuilder::NeScalar(const NodePtr &input, const NodePtr &other) { return NativeFunc::NeScalar(input, other); }

NodePtr FuncBuilder::Conv2DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
  return NativeFunc::Conv2DExt(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::RandnLike(const NodePtr &input, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                               const NodePtr &device) {
  return NativeFunc::RandnLike(input, seed, offset, dtype, device);
}

NodePtr FuncBuilder::Conv3DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                   const NodePtr &groups) {
  return NativeFunc::Conv3DPadding(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::Ceil(const NodePtr &input) { return NativeFunc::Ceil(input); }

NodePtr FuncBuilder::EluGradExt(const NodePtr &dout, const NodePtr &x_or_out, const NodePtr &alpha,
                                const NodePtr &is_result) {
  return NativeFunc::EluGradExt(dout, x_or_out, alpha, is_result);
}

NodePtr FuncBuilder::TypeAs(const NodePtr &input, const NodePtr &other) { return NativeFunc::TypeAs(input, other); }

NodePtr FuncBuilder::BatchNormStats(const NodePtr &input, const NodePtr &eps) {
  return NativeFunc::BatchNormStats(input, eps);
}

NodePtr FuncBuilder::MaxDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::MaxDim(input, dim, keepdim);
}

NodePtr FuncBuilder::FFNExt(const NodePtr &x, const NodePtr &weight1, const NodePtr &weight2,
                            const NodePtr &expertTokens, const NodePtr &bias1, const NodePtr &bias2,
                            const NodePtr &scale, const NodePtr &offset, const NodePtr &deqScale1,
                            const NodePtr &deqScale2, const NodePtr &antiquant_scale1, const NodePtr &antiquant_scale2,
                            const NodePtr &antiquant_offset1, const NodePtr &antiquant_offset2,
                            const NodePtr &activation, const NodePtr &inner_precise) {
  return NativeFunc::FFNExt(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2,
                            antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, activation,
                            inner_precise);
}

NodePtr FuncBuilder::ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                     const NodePtr &bias, const NodePtr &stride, const NodePtr &padding,
                                     const NodePtr &dilation, const NodePtr &transposed, const NodePtr &output_padding,
                                     const NodePtr &groups, const NodePtr &output_mask) {
  return NativeFunc::ConvolutionGrad(dout, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                     groups, output_mask);
}

NodePtr FuncBuilder::MSELossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
  return NativeFunc::MSELossExt(input, target, reduction);
}

NodePtr FuncBuilder::NLLLoss2dGrad(const NodePtr &loss_grad, const NodePtr &input, const NodePtr &target,
                                   const NodePtr &weight, const NodePtr &reduction, const NodePtr &ignore_index,
                                   const NodePtr &total_weight) {
  return NativeFunc::NLLLoss2dGrad(loss_grad, input, target, weight, reduction, ignore_index, total_weight);
}

NodePtr FuncBuilder::ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad1DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::AdaptiveAvgPool3DExt(const NodePtr &input, const NodePtr &output_size) {
  return NativeFunc::AdaptiveAvgPool3DExt(input, output_size);
}

NodePtr FuncBuilder::PromptFlashAttention(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &attn_mask,
  const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv, const NodePtr &pse_shift,
  const NodePtr &deq_scale1, const NodePtr &quant_scale1, const NodePtr &deq_scale2, const NodePtr &quant_scale2,
  const NodePtr &quant_offset2, const NodePtr &num_heads, const NodePtr &scale_value, const NodePtr &pre_tokens,
  const NodePtr &next_tokens, const NodePtr &input_layout, const NodePtr &num_key_value_heads,
  const NodePtr &sparse_mode, const NodePtr &inner_precise) {
  return NativeFunc::PromptFlashAttention(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                                          pse_shift, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2,
                                          num_heads, scale_value, pre_tokens, next_tokens, input_layout,
                                          num_key_value_heads, sparse_mode, inner_precise);
}

NodePtr FuncBuilder::MinDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::MinDim(input, dim, keepdim);
}

NodePtr FuncBuilder::MatmulReduceScatter(const NodePtr &input, const NodePtr &x2, const NodePtr &group,
                                         const NodePtr &world_size, const NodePtr &reduce_op, const NodePtr &bias,
                                         const NodePtr &comm_turn, const NodePtr &trans_input,
                                         const NodePtr &trans_x2) {
  return NativeFunc::MatmulReduceScatter(input, x2, group, world_size, reduce_op, bias, comm_turn, trans_input,
                                         trans_x2);
}

NodePtr FuncBuilder::SpeedFusionAttentionGrad(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy, const NodePtr &head_num,
  const NodePtr &input_layout, const NodePtr &pse, const NodePtr &padding_mask, const NodePtr &atten_mask,
  const NodePtr &softmax_max, const NodePtr &softmax_sum, const NodePtr &softmax_in, const NodePtr &attention_in,
  const NodePtr &scale_value, const NodePtr &keep_prob, const NodePtr &pre_tokens, const NodePtr &next_tokens,
  const NodePtr &inner_precise, const NodePtr &seed, const NodePtr &offset, const NodePtr &numels,
  const NodePtr &prefix, const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen, const NodePtr &sparse_mode,
  const NodePtr &gen_mask_parallel, const NodePtr &sync, const NodePtr &pse_type, const NodePtr &q_start_idx,
  const NodePtr &kv_start_idx) {
  return NativeFunc::SpeedFusionAttentionGrad(
    query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
    attention_in, scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix,
    actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}

NodePtr FuncBuilder::MaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
  return NativeFunc::MaskedFillScalar(input, mask, value);
}

NodePtr FuncBuilder::Atan2Ext(const NodePtr &input, const NodePtr &other) { return NativeFunc::Atan2Ext(input, other); }

NodePtr FuncBuilder::DequantSwigluQuant(const NodePtr &x, const NodePtr &weight_scale, const NodePtr &activation_scale,
                                        const NodePtr &bias, const NodePtr &quant_scale, const NodePtr &quant_offset,
                                        const NodePtr &group_index, const NodePtr &activate_left,
                                        const NodePtr &quant_mode) {
  return NativeFunc::DequantSwigluQuant(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
                                        activate_left, quant_mode);
}

NodePtr FuncBuilder::InplaceSiLU(const NodePtr &input) { return NativeFunc::InplaceSiLU(input); }

NodePtr FuncBuilder::Var(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
  return NativeFunc::Var(input, dim, correction, keepdim);
}

NodePtr FuncBuilder::Mv(const NodePtr &input, const NodePtr &vec) { return NativeFunc::Mv(input, vec); }

NodePtr FuncBuilder::AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v,
                           const NodePtr &gradient, const NodePtr &step, const NodePtr &lr, const NodePtr &beta1,
                           const NodePtr &beta2, const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad,
                           const NodePtr &maximize) {
  return NativeFunc::AdamW(var, m, v, max_v, gradient, step, lr, beta1, beta2, decay, eps, amsgrad, maximize);
}

NodePtr FuncBuilder::InplaceMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &C) {
  return NativeFunc::InplaceMatmulAdd(x, weight, C);
}

NodePtr FuncBuilder::BincountExt(const NodePtr &input, const NodePtr &weights, const NodePtr &minlength) {
  return NativeFunc::BincountExt(input, weights, minlength);
}

NodePtr FuncBuilder::SeluGrad(const NodePtr &gradient, const NodePtr &result) {
  return NativeFunc::SeluGrad(gradient, result);
}

NodePtr FuncBuilder::NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                        const NodePtr &offset) {
  return NativeFunc::NormalTensorTensor(mean, std, seed, offset);
}

NodePtr FuncBuilder::ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceAll(input, axis, keep_dims);
}

NodePtr FuncBuilder::DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
  return NativeFunc::DropoutDoMaskExt(input, mask, p);
}

NodePtr FuncBuilder::UpsampleBicubic2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleBicubic2DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::AddcmulExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                                const NodePtr &value) {
  return NativeFunc::AddcmulExt(input, tensor1, tensor2, value);
}

NodePtr FuncBuilder::InplaceScatterValueReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                               const NodePtr &value, const NodePtr &reduce) {
  return NativeFunc::InplaceScatterValueReduce(input, dim, index, value, reduce);
}

NodePtr FuncBuilder::Gcd(const NodePtr &input, const NodePtr &other) { return NativeFunc::Gcd(input, other); }

NodePtr FuncBuilder::Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) {
  return NativeFunc::Eye(n, m, dtype);
}

NodePtr FuncBuilder::NanToNum(const NodePtr &input, const NodePtr &nan, const NodePtr &posinf, const NodePtr &neginf) {
  return NativeFunc::NanToNum(input, nan, posinf, neginf);
}

NodePtr FuncBuilder::GeluExt(const NodePtr &input, const NodePtr &approximate) {
  return NativeFunc::GeluExt(input, approximate);
}

NodePtr FuncBuilder::Repeat(const NodePtr &input, const NodePtr &repeats) { return NativeFunc::Repeat(input, repeats); }

NodePtr FuncBuilder::Conv2DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                   const NodePtr &groups) {
  return NativeFunc::Conv2DPadding(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::InplaceSubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::InplaceSubScalar(input, other, alpha);
}

NodePtr FuncBuilder::Copy(const NodePtr &input) { return NativeFunc::Copy(input); }

NodePtr FuncBuilder::Zeros(const NodePtr &size, const NodePtr &dtype) { return NativeFunc::Zeros(size, dtype); }

NodePtr FuncBuilder::Muls(const NodePtr &input, const NodePtr &other) { return NativeFunc::Muls(input, other); }

NodePtr FuncBuilder::NLLLossGrad(const NodePtr &logits, const NodePtr &loss_grad, const NodePtr &labels,
                                 const NodePtr &weight, const NodePtr &total_weight, const NodePtr &reduction,
                                 const NodePtr &ignore_index) {
  return NativeFunc::NLLLossGrad(logits, loss_grad, labels, weight, total_weight, reduction, ignore_index);
}

NodePtr FuncBuilder::AdaptiveAvgPool1D(const NodePtr &input, const NodePtr &output_size) {
  return NativeFunc::AdaptiveAvgPool1D(input, output_size);
}

NodePtr FuncBuilder::Index(const NodePtr &input, const NodePtr &indices) { return NativeFunc::Index(input, indices); }

NodePtr FuncBuilder::HardtanhGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &min_val,
                                  const NodePtr &max_val) {
  return NativeFunc::HardtanhGrad(dout, input, min_val, max_val);
}

NodePtr FuncBuilder::RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                         const NodePtr &output_size) {
  return NativeFunc::RepeatInterleaveInt(input, repeats, dim, output_size);
}

NodePtr FuncBuilder::Conv3DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
  return NativeFunc::Conv3DExt(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::Sigmoid(const NodePtr &input) { return NativeFunc::Sigmoid(input); }

NodePtr FuncBuilder::Threshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) {
  return NativeFunc::Threshold(input, threshold, value);
}

NodePtr FuncBuilder::NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                                      const NodePtr &offset) {
  return NativeFunc::NormalFloatFloat(mean, std, size, seed, offset);
}

NodePtr FuncBuilder::BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                  const NodePtr &running_mean, const NodePtr &runnning_var, const NodePtr &training,
                                  const NodePtr &momentum, const NodePtr &epsilon) {
  return NativeFunc::BatchNormExt(input, weight, bias, running_mean, runnning_var, training, momentum, epsilon);
}

NodePtr FuncBuilder::AsinExt(const NodePtr &input) { return NativeFunc::AsinExt(input); }

NodePtr FuncBuilder::Cast(const NodePtr &input, const NodePtr &dtype) { return NativeFunc::Cast(input, dtype); }

NodePtr FuncBuilder::LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape,
                                      const NodePtr &mean, const NodePtr &variance, const NodePtr &gamma,
                                      const NodePtr &beta, const NodePtr &output_mask) {
  return NativeFunc::LayerNormGradExt(dy, x, normalized_shape, mean, variance, gamma, beta, output_mask);
}

NodePtr FuncBuilder::KLDivGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                               const NodePtr &reduction, const NodePtr &log_target) {
  return NativeFunc::KLDivGrad(grad_output, input, target, reduction, log_target);
}

NodePtr FuncBuilder::ApplyRotaryPosEmb(const NodePtr &query, const NodePtr &key, const NodePtr &cos, const NodePtr &sin,
                                       const NodePtr &position_ids, const NodePtr &cos_format) {
  return NativeFunc::ApplyRotaryPosEmb(query, key, cos, sin, position_ids, cos_format);
}

NodePtr FuncBuilder::BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                                 const NodePtr &transpose_b) {
  return NativeFunc::BatchMatMul(x, y, transpose_a, transpose_b);
}

NodePtr FuncBuilder::HSigmoid(const NodePtr &input) { return NativeFunc::HSigmoid(input); }

NodePtr FuncBuilder::NonZero(const NodePtr &input) { return NativeFunc::NonZero(input); }

NodePtr FuncBuilder::Meshgrid(const NodePtr &inputs, const NodePtr &indexing) {
  return NativeFunc::Meshgrid(inputs, indexing);
}

NodePtr FuncBuilder::Erfinv(const NodePtr &input) { return NativeFunc::Erfinv(input); }

NodePtr FuncBuilder::MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask,
                                         const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                         const NodePtr &dilation, const NodePtr &ceil_mode,
                                         const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolGradWithMask(x, grad, mask, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                                const NodePtr &offset) {
  return NativeFunc::UniformExt(tensor, a, b, seed, offset);
}

NodePtr FuncBuilder::GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                   const NodePtr &padding_mode, const NodePtr &align_corners) {
  return NativeFunc::GridSampler2D(input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::RemainderTensorTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::RemainderTensorTensor(input, other);
}

NodePtr FuncBuilder::Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) {
  return NativeFunc::Dense(input, weight, bias);
}

NodePtr FuncBuilder::SeLUExt(const NodePtr &input) { return NativeFunc::SeLUExt(input); }

NodePtr FuncBuilder::AsinhExt(const NodePtr &input) { return NativeFunc::AsinhExt(input); }

NodePtr FuncBuilder::AcosExt(const NodePtr &input) { return NativeFunc::AcosExt(input); }

NodePtr FuncBuilder::SoftMarginLoss(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
  return NativeFunc::SoftMarginLoss(input, target, reduction);
}

NodePtr FuncBuilder::ChunkView(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
  return NativeFunc::ChunkView(input, chunks, dim);
}

NodePtr FuncBuilder::InplaceMuls(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceMuls(input, other);
}

NodePtr FuncBuilder::HSwish(const NodePtr &input) { return NativeFunc::HSwish(input); }

NodePtr FuncBuilder::TExt(const NodePtr &input) { return NativeFunc::TExt(input); }

NodePtr FuncBuilder::UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                        const NodePtr &align_corners) {
  return NativeFunc::UpsampleBilinear2D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::Cross(const NodePtr &input, const NodePtr &other, const NodePtr &dim) {
  return NativeFunc::Cross(input, other, dim);
}

NodePtr FuncBuilder::SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
  return NativeFunc::SumExt(input, dim, keepdim, dtype);
}

NodePtr FuncBuilder::InplacePut(const NodePtr &input, const NodePtr &index, const NodePtr &source,
                                const NodePtr &accumulate) {
  return NativeFunc::InplacePut(input, index, source, accumulate);
}

NodePtr FuncBuilder::SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                              const NodePtr &step) {
  return NativeFunc::SliceExt(input, dim, start, end, step);
}

NodePtr FuncBuilder::ScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                                  const NodePtr &reduce) {
  return NativeFunc::ScatterValue(input, dim, index, src, reduce);
}

NodePtr FuncBuilder::ReverseV2(const NodePtr &input, const NodePtr &axis) { return NativeFunc::ReverseV2(input, axis); }

NodePtr FuncBuilder::UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest2DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::Nansum(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
  return NativeFunc::Nansum(input, dim, keepdim, dtype);
}

NodePtr FuncBuilder::UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest1DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::Maximum(const NodePtr &input, const NodePtr &other) { return NativeFunc::Maximum(input, other); }

NodePtr FuncBuilder::MoeTokenPermuteGrad(const NodePtr &permuted_tokens_grad, const NodePtr &sorted_indices,
                                         const NodePtr &num_topk, const NodePtr &padded_mode) {
  return NativeFunc::MoeTokenPermuteGrad(permuted_tokens_grad, sorted_indices, num_topk, padded_mode);
}

NodePtr FuncBuilder::DivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
  return NativeFunc::DivMods(input, other, rounding_mode);
}

NodePtr FuncBuilder::Trunc(const NodePtr &input) { return NativeFunc::Trunc(input); }

NodePtr FuncBuilder::MedianDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::MedianDim(input, dim, keepdim);
}

NodePtr FuncBuilder::Max(const NodePtr &input) { return NativeFunc::Max(input); }

NodePtr FuncBuilder::MedianExt(const NodePtr &input) { return NativeFunc::MedianExt(input); }

NodePtr FuncBuilder::Erfc(const NodePtr &input) { return NativeFunc::Erfc(input); }

NodePtr FuncBuilder::GLU(const NodePtr &x, const NodePtr &axis) { return NativeFunc::GLU(x, axis); }

NodePtr FuncBuilder::Reciprocal(const NodePtr &input) { return NativeFunc::Reciprocal(input); }

NodePtr FuncBuilder::SoftShrink(const NodePtr &input, const NodePtr &lambd) {
  return NativeFunc::SoftShrink(input, lambd);
}

NodePtr FuncBuilder::InplaceRemainderTensorScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceRemainderTensorScalar(input, other);
}

NodePtr FuncBuilder::Contiguous(const NodePtr &input) { return NativeFunc::Contiguous(input); }

NodePtr FuncBuilder::ToDtype(const NodePtr &input, const NodePtr &dtype, const NodePtr &non_blocking,
                             const NodePtr &copy) {
  return NativeFunc::ToDtype(input, dtype, non_blocking, copy);
}

NodePtr FuncBuilder::SplitWithSize(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
  return NativeFunc::SplitWithSize(input, split_size, dim);
}

NodePtr FuncBuilder::MoeTokenPermute(const NodePtr &tokens, const NodePtr &indices, const NodePtr &num_out_tokens,
                                     const NodePtr &padded_mode) {
  return NativeFunc::MoeTokenPermute(tokens, indices, num_out_tokens, padded_mode);
}

NodePtr FuncBuilder::AdaptiveMaxPool2D(const NodePtr &input, const NodePtr &output_size) {
  return NativeFunc::AdaptiveMaxPool2D(input, output_size);
}

NodePtr FuncBuilder::ReplicationPad3D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad3D(input, padding);
}

NodePtr FuncBuilder::SilentCheckV3(const NodePtr &val, const NodePtr &max, const NodePtr &avg,
                                   const NodePtr &input_grad, const NodePtr &step, const NodePtr &c_thresh_l1,
                                   const NodePtr &c_thresh_l2, const NodePtr &beta1, const NodePtr &npu_asd_detect) {
  return NativeFunc::SilentCheckV3(val, max, avg, input_grad, step, c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect);
}

NodePtr FuncBuilder::BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                        const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropy(input, target, weight, reduction);
}

NodePtr FuncBuilder::L1LossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
  return NativeFunc::L1LossExt(input, target, reduction);
}

NodePtr FuncBuilder::Min(const NodePtr &input) { return NativeFunc::Min(input); }

NodePtr FuncBuilder::InplaceBernoulliScalar(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                            const NodePtr &offset) {
  return NativeFunc::InplaceBernoulliScalar(input, p, seed, offset);
}

NodePtr FuncBuilder::FloorDivScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::FloorDivScalar(input, other);
}

NodePtr FuncBuilder::FullLike(const NodePtr &input, const NodePtr &fill_value, const NodePtr &dtype) {
  return NativeFunc::FullLike(input, fill_value, dtype);
}

NodePtr FuncBuilder::Empty(const NodePtr &size, const NodePtr &dtype, const NodePtr &device,
                           const NodePtr &pin_memory) {
  return NativeFunc::Empty(size, dtype, device, pin_memory);
}

NodePtr FuncBuilder::MultiScaleDeformableAttnGrad(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                                  const NodePtr &locations_trans, const NodePtr &weight,
                                                  const NodePtr &grad_output) {
  return NativeFunc::MultiScaleDeformableAttnGrad(value, shape, offset, locations_trans, weight, grad_output);
}

NodePtr FuncBuilder::LogSoftmaxGrad(const NodePtr &logits, const NodePtr &grad, const NodePtr &axis) {
  return NativeFunc::LogSoftmaxGrad(logits, grad, axis);
}

NodePtr FuncBuilder::RandInt(const NodePtr &low, const NodePtr &high, const NodePtr &shape, const NodePtr &seed,
                             const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) {
  return NativeFunc::RandInt(low, high, shape, seed, offset, dtype, device);
}

NodePtr FuncBuilder::Frac(const NodePtr &input) { return NativeFunc::Frac(input); }

NodePtr FuncBuilder::ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::ArgMaxExt(input, dim, keepdim);
}

NodePtr FuncBuilder::UniqueConsecutive(const NodePtr &input, const NodePtr &return_inverse,
                                       const NodePtr &return_counts, const NodePtr &dim) {
  return NativeFunc::UniqueConsecutive(input, return_inverse, return_counts, dim);
}

NodePtr FuncBuilder::ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceAny(x, axis, keep_dims);
}

NodePtr FuncBuilder::UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                          const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleLinear1DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::InplaceHardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) {
  return NativeFunc::InplaceHardtanh(input, min_val, max_val);
}

NodePtr FuncBuilder::IndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                     const NodePtr &value) {
  return NativeFunc::IndexFillScalar(input, dim, index, value);
}

NodePtr FuncBuilder::PagedAttention(const NodePtr &query, const NodePtr &key_cache, const NodePtr &value_cache,
                                    const NodePtr &block_tables, const NodePtr &context_lens,
                                    const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                                    const NodePtr &attn_mask, const NodePtr &q_seq_lens, const NodePtr &alibi_mask,
                                    const NodePtr &head_num, const NodePtr &scale_value, const NodePtr &kv_head_num,
                                    const NodePtr &kv_cache_quant_mode, const NodePtr &mask_mode,
                                    const NodePtr &mla_v_dim) {
  return NativeFunc::PagedAttention(query, key_cache, value_cache, block_tables, context_lens, antiquant_scale,
                                    antiquant_offset, attn_mask, q_seq_lens, alibi_mask, head_num, scale_value,
                                    kv_head_num, kv_cache_quant_mode, mask_mode, mla_v_dim);
}

NodePtr FuncBuilder::PowTensorScalar(const NodePtr &input, const NodePtr &exponent) {
  return NativeFunc::PowTensorScalar(input, exponent);
}

NodePtr FuncBuilder::NonZeroExt(const NodePtr &input) { return NativeFunc::NonZeroExt(input); }

NodePtr FuncBuilder::SoftMarginLossGrad(const NodePtr &predict, const NodePtr &label, const NodePtr &dout,
                                        const NodePtr &reduction) {
  return NativeFunc::SoftMarginLossGrad(predict, label, dout, reduction);
}

NodePtr FuncBuilder::SelectV2(const NodePtr &condition, const NodePtr &input, const NodePtr &other) {
  return NativeFunc::SelectV2(condition, input, other);
}

NodePtr FuncBuilder::ReluGrad(const NodePtr &y_backprop, const NodePtr &x) {
  return NativeFunc::ReluGrad(y_backprop, x);
}

NodePtr FuncBuilder::EluExt(const NodePtr &input, const NodePtr &alpha) { return NativeFunc::EluExt(input, alpha); }

NodePtr FuncBuilder::IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
  return NativeFunc::IndexSelect(input, dim, index);
}

NodePtr FuncBuilder::Split(const NodePtr &input_x, const NodePtr &axis, const NodePtr &output_num) {
  return NativeFunc::Split(input_x, axis, output_num);
}

NodePtr FuncBuilder::IndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &source,
                                 const NodePtr &alpha) {
  return NativeFunc::IndexAddExt(input, dim, index, source, alpha);
}

NodePtr FuncBuilder::DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::DropoutExt(input, p, seed, offset);
}

NodePtr FuncBuilder::SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                                     const NodePtr &threshold) {
  return NativeFunc::SoftplusGradExt(dout, x, beta, threshold);
}

NodePtr FuncBuilder::IsFinite(const NodePtr &input) { return NativeFunc::IsFinite(input); }

NodePtr FuncBuilder::Abs(const NodePtr &input) { return NativeFunc::Abs(input); }

NodePtr FuncBuilder::NLLLoss(const NodePtr &logits, const NodePtr &labels, const NodePtr &weight,
                             const NodePtr &reduction, const NodePtr &ignore_index) {
  return NativeFunc::NLLLoss(logits, labels, weight, reduction, ignore_index);
}

NodePtr FuncBuilder::UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                             const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleTrilinear3DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) {
  return NativeFunc::RmsNormGrad(dy, x, rstd, gamma);
}

NodePtr FuncBuilder::LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                                      const NodePtr &is_result) {
  return NativeFunc::LeakyReLUGradExt(dy, input, negative_slope, is_result);
}

NodePtr FuncBuilder::LogSumExp(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::LogSumExp(input, dim, keepdim);
}

NodePtr FuncBuilder::Erf(const NodePtr &input) { return NativeFunc::Erf(input); }

NodePtr FuncBuilder::SilentCheckV2(const NodePtr &val, const NodePtr &input_grad, const NodePtr &sfda,
                                   const NodePtr &step, const NodePtr &c_min_steps, const NodePtr &c_thresh_l1,
                                   const NodePtr &c_coeff_l1, const NodePtr &c_thresh_l2, const NodePtr &c_coeff_l2,
                                   const NodePtr &npu_asd_detect) {
  return NativeFunc::SilentCheckV2(val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2,
                                   c_coeff_l2, npu_asd_detect);
}

NodePtr FuncBuilder::InplaceScatterSrc(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                       const NodePtr &src) {
  return NativeFunc::InplaceScatterSrc(input, dim, index, src);
}

NodePtr FuncBuilder::BitwiseAndScalar(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::BitwiseAndScalar(input, other);
}

NodePtr FuncBuilder::MSELossGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &target,
                                    const NodePtr &reduction) {
  return NativeFunc::MSELossGradExt(dout, x, target, reduction);
}

NodePtr FuncBuilder::UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                      const NodePtr &align_corners) {
  return NativeFunc::UpsampleLinear1D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::ReduceMin(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceMin(x, axis, keep_dims);
}

NodePtr FuncBuilder::LogicalNot(const NodePtr &input) { return NativeFunc::LogicalNot(input); }

NodePtr FuncBuilder::SoftShrinkGrad(const NodePtr &input_grad, const NodePtr &input_x, const NodePtr &lambd) {
  return NativeFunc::SoftShrinkGrad(input_grad, input_x, lambd);
}

NodePtr FuncBuilder::CrossEntropyLossGrad(const NodePtr &grad_loss, const NodePtr &log_prob, const NodePtr &target,
                                          const NodePtr &weight, const NodePtr &grad_zloss,
                                          const NodePtr &lse_for_zloss, const NodePtr &reduction,
                                          const NodePtr &ignore_index, const NodePtr &label_smoothing,
                                          const NodePtr &lse_square_scale_for_zloss) {
  return NativeFunc::CrossEntropyLossGrad(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction,
                                          ignore_index, label_smoothing, lse_square_scale_for_zloss);
}

NodePtr FuncBuilder::MatMul(const NodePtr &input, const NodePtr &mat2, const NodePtr &transpose_a,
                            const NodePtr &transpose_b) {
  return NativeFunc::MatMul(input, mat2, transpose_a, transpose_b);
}

NodePtr FuncBuilder::Triu(const NodePtr &input, const NodePtr &diagonal) { return NativeFunc::Triu(input, diagonal); }

NodePtr FuncBuilder::Lerp(const NodePtr &input, const NodePtr &end, const NodePtr &weight) {
  return NativeFunc::Lerp(input, end, weight);
}

NodePtr FuncBuilder::ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad2DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::InplaceDivs(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceDivs(input, other);
}

NodePtr FuncBuilder::Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                               const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Im2ColExt(input, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::DiagExt(const NodePtr &input, const NodePtr &diagonal) {
  return NativeFunc::DiagExt(input, diagonal);
}

NodePtr FuncBuilder::InplaceFillTensor(const NodePtr &input, const NodePtr &value) {
  return NativeFunc::InplaceFillTensor(input, value);
}

NodePtr FuncBuilder::NewFull(const NodePtr &input, const NodePtr &size, const NodePtr &fill_value,
                             const NodePtr &dtype) {
  return NativeFunc::NewFull(input, size, fill_value, dtype);
}

NodePtr FuncBuilder::PReLU(const NodePtr &input, const NodePtr &weight) { return NativeFunc::PReLU(input, weight); }

NodePtr FuncBuilder::IndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                     const NodePtr &value) {
  return NativeFunc::IndexFillTensor(input, dim, index, value);
}

NodePtr FuncBuilder::ConvTranspose2D(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                     const NodePtr &stride, const NodePtr &padding, const NodePtr &output_padding,
                                     const NodePtr &groups, const NodePtr &dilation) {
  return NativeFunc::ConvTranspose2D(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

NodePtr FuncBuilder::InplaceRemainderTensorTensor(const NodePtr &input, const NodePtr &other) {
  return NativeFunc::InplaceRemainderTensorTensor(input, other);
}

NodePtr FuncBuilder::Sinc(const NodePtr &input) { return NativeFunc::Sinc(input); }

NodePtr FuncBuilder::InplaceAddsExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::InplaceAddsExt(input, other, alpha);
}

NodePtr FuncBuilder::Tan(const NodePtr &input) { return NativeFunc::Tan(input); }

NodePtr FuncBuilder::UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest1D(x, output_size, scales);
}

NodePtr FuncBuilder::MoeDistributeCombine(
  const NodePtr &expand_x, const NodePtr &expert_ids, const NodePtr &expand_idx, const NodePtr &ep_send_counts,
  const NodePtr &expert_scales, const NodePtr &ep_world_size, const NodePtr &ep_rank_id, const NodePtr &moe_expert_num,
  const NodePtr &tp_send_counts, const NodePtr &x_active_mask, const NodePtr &activate_scale,
  const NodePtr &weight_scale, const NodePtr &group_list, const NodePtr &expand_scales, const NodePtr &group_ep,
  const NodePtr &group_tp, const NodePtr &tp_world_size, const NodePtr &tp_rank_id, const NodePtr &expert_shard_type,
  const NodePtr &shared_expert_num, const NodePtr &shared_export_rank_num, const NodePtr &global_bs,
  const NodePtr &out_dtype, const NodePtr &common_quant_mode, const NodePtr &group_list_type) {
  return NativeFunc::MoeDistributeCombine(
    expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, ep_world_size, ep_rank_id, moe_expert_num,
    tp_send_counts, x_active_mask, activate_scale, weight_scale, group_list, expand_scales, group_ep, group_tp,
    tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_export_rank_num, global_bs, out_dtype,
    common_quant_mode, group_list_type);
}

NodePtr FuncBuilder::ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) {
  return NativeFunc::ConstantPadND(input, padding, value);
}

NodePtr FuncBuilder::UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest3D(x, output_size, scales);
}

NodePtr FuncBuilder::Rsqrt(const NodePtr &input) { return NativeFunc::Rsqrt(input); }

NodePtr FuncBuilder::RingAttentionUpdate(const NodePtr &prev_attn_out, const NodePtr &prev_softmax_max,
                                         const NodePtr &prev_softmax_sum, const NodePtr &cur_attn_out,
                                         const NodePtr &cur_softmax_max, const NodePtr &cur_softmax_sum,
                                         const NodePtr &actual_seq_qlen, const NodePtr &layout) {
  return NativeFunc::RingAttentionUpdate(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout);
}

NodePtr FuncBuilder::InplaceMaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
  return NativeFunc::InplaceMaskedFillScalar(input, mask, value);
}

NodePtr FuncBuilder::NewEmpty(const NodePtr &input, const NodePtr &size, const NodePtr &dtype, const NodePtr &device) {
  return NativeFunc::NewEmpty(input, size, dtype, device);
}

NodePtr FuncBuilder::CrossEntropyLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                      const NodePtr &reduction, const NodePtr &ignore_index,
                                      const NodePtr &label_smoothing, const NodePtr &lse_square_scale_for_zloss,
                                      const NodePtr &return_zloss) {
  return NativeFunc::CrossEntropyLoss(input, target, weight, reduction, ignore_index, label_smoothing,
                                      lse_square_scale_for_zloss, return_zloss);
}

NodePtr FuncBuilder::AddScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::AddScalar(input, other, alpha);
}

NodePtr FuncBuilder::UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                            const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleBilinear2DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::Floor(const NodePtr &input) { return NativeFunc::Floor(input); }

NodePtr FuncBuilder::Mla(const NodePtr &query, const NodePtr &q_rope, const NodePtr &kv_cache, const NodePtr &k_rope,
                         const NodePtr &block_tables, const NodePtr &attn_mask, const NodePtr &deq_scale_qk,
                         const NodePtr &deq_scale_pv, const NodePtr &q_seq_lens, const NodePtr &context_lens,
                         const NodePtr &head_num, const NodePtr &scale_value, const NodePtr &kv_head_num,
                         const NodePtr &mask_mode, const NodePtr &is_ring) {
  return NativeFunc::Mla(query, q_rope, kv_cache, k_rope, block_tables, attn_mask, deq_scale_qk, deq_scale_pv,
                         q_seq_lens, context_lens, head_num, scale_value, kv_head_num, mask_mode, is_ring);
}

NodePtr FuncBuilder::MaskedSelect(const NodePtr &input, const NodePtr &mask) {
  return NativeFunc::MaskedSelect(input, mask);
}

NodePtr FuncBuilder::NarrowView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) {
  return NativeFunc::NarrowView(input, dim, start, length);
}

NodePtr FuncBuilder::Sinh(const NodePtr &input) { return NativeFunc::Sinh(input); }

NodePtr FuncBuilder::Conv1DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                   const NodePtr &groups) {
  return NativeFunc::Conv1DPadding(input, weight, bias, stride, padding, dilation, groups);
}

NodePtr FuncBuilder::QuantMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                                 const NodePtr &pertoken_scale, const NodePtr &bias, const NodePtr &output_dtype,
                                 const NodePtr &x1_dtype, const NodePtr &x2_dtype, const NodePtr &pertoken_scale_dtype,
                                 const NodePtr &scale_dtype, const NodePtr &group_sizes) {
  return NativeFunc::QuantMatmul(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, x1_dtype, x2_dtype,
                                 pertoken_scale_dtype, scale_dtype, group_sizes);
}

NodePtr FuncBuilder::AddRmsNormQuantV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &scale,
                                       const NodePtr &offset, const NodePtr &epsilon) {
  return NativeFunc::AddRmsNormQuantV2(x1, x2, gamma, scale, offset, epsilon);
}

NodePtr FuncBuilder::MoeInitRouting(const NodePtr &x, const NodePtr &row_idx, const NodePtr &expert_idx,
                                    const NodePtr &active_num) {
  return NativeFunc::MoeInitRouting(x, row_idx, expert_idx, active_num);
}

NodePtr FuncBuilder::FusedInferAttentionScore(
  const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &pse_shift, const NodePtr &attn_mask,
  const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv, const NodePtr &dequant_scale1,
  const NodePtr &quant_scale1, const NodePtr &dequant_scale2, const NodePtr &quant_scale2, const NodePtr &quant_offset2,
  const NodePtr &antiquant_scale, const NodePtr &antiquant_offset, const NodePtr &block_table,
  const NodePtr &query_padding_size, const NodePtr &kv_padding_size, const NodePtr &key_antiquant_scale,
  const NodePtr &key_antiquant_offset, const NodePtr &value_antiquant_scale, const NodePtr &value_antiquant_offset,
  const NodePtr &key_shared_prefix, const NodePtr &value_shared_prefix, const NodePtr &actual_shared_prefix_len,
  const NodePtr &num_heads, const NodePtr &scale_value, const NodePtr &pre_tokens, const NodePtr &next_tokens,
  const NodePtr &input_layout, const NodePtr &num_key_value_heads, const NodePtr &sparse_mode,
  const NodePtr &inner_precise, const NodePtr &block_size, const NodePtr &antiquant_mode,
  const NodePtr &softmax_lse_flag, const NodePtr &key_antiquant_mode, const NodePtr &value_antiquant_mode) {
  return NativeFunc::FusedInferAttentionScore(
    query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1,
    dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, query_padding_size,
    kv_padding_size, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset,
    key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, num_heads, scale_value, pre_tokens, next_tokens,
    input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag,
    key_antiquant_mode, value_antiquant_mode);
}

NodePtr FuncBuilder::GroupedMatmulV4(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                     const NodePtr &offset, const NodePtr &antiquant_scale,
                                     const NodePtr &antiquant_offset, const NodePtr &pre_token_scale,
                                     const NodePtr &group_list, const NodePtr &activation_input,
                                     const NodePtr &activation_quant_scale, const NodePtr &activation_quant_offset,
                                     const NodePtr &split_item, const NodePtr &group_type,
                                     const NodePtr &group_list_type, const NodePtr &act_type,
                                     const NodePtr &output_dtype) {
  return NativeFunc::GroupedMatmulV4(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pre_token_scale,
                                     group_list, activation_input, activation_quant_scale, activation_quant_offset,
                                     split_item, group_type, group_list_type, act_type, output_dtype);
}

NodePtr FuncBuilder::QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                                      const NodePtr &bias, const NodePtr &pertokenScaleOptional,
                                      const NodePtr &transpose_x1, const NodePtr &transpose_x2, const NodePtr &dtype) {
  return NativeFunc::QuantBatchMatmul(x1, x2, scale, offset, bias, pertokenScaleOptional, transpose_x1, transpose_x2,
                                      dtype);
}

NodePtr FuncBuilder::MoeComputeExpertTokens(const NodePtr &sorted_experts, const NodePtr &num_expert) {
  return NativeFunc::MoeComputeExpertTokens(sorted_experts, num_expert);
}

NodePtr FuncBuilder::GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                   const NodePtr &offset, const NodePtr &antiquant_scale,
                                   const NodePtr &antiquant_offset, const NodePtr &group_list,
                                   const NodePtr &split_item, const NodePtr &group_type, const NodePtr &transpose_a,
                                   const NodePtr &transpose_b) {
  return NativeFunc::GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                   split_item, group_type, transpose_a, transpose_b);
}

NodePtr FuncBuilder::WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                            const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                            const NodePtr &quant_offset, const NodePtr &bias,
                                            const NodePtr &transpose_x, const NodePtr &transpose_weight,
                                            const NodePtr &antiquant_group_size) {
  return NativeFunc::WeightQuantBatchMatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset,
                                            bias, transpose_x, transpose_weight, antiquant_group_size);
}

NodePtr FuncBuilder::MatmulAllReduceAddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &bias,
                                               const NodePtr &residual, const NodePtr &gamma, const NodePtr &epsilon,
                                               const NodePtr &group, const NodePtr &reduce_op, const NodePtr &comm_turn,
                                               const NodePtr &stream_mode) {
  return NativeFunc::MatmulAllReduceAddRmsNorm(x1, x2, bias, residual, gamma, epsilon, group, reduce_op, comm_turn,
                                               stream_mode);
}

NodePtr FuncBuilder::GroupedMatmulV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                     const NodePtr &offset, const NodePtr &antiquant_scale,
                                     const NodePtr &antiquant_offset, const NodePtr &group_list,
                                     const NodePtr &split_item, const NodePtr &group_type) {
  return NativeFunc::GroupedMatmulV2(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                     split_item, group_type);
}

NodePtr FuncBuilder::QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                             const NodePtr &rounding_mode, const NodePtr &dst_type) {
  return NativeFunc::QuantV2(x, scale, offset, sqrt_mode, rounding_mode, dst_type);
}

NodePtr FuncBuilder::MoeInitRoutingV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                                      const NodePtr &expert_capacity, const NodePtr &expert_num,
                                      const NodePtr &drop_pad_mode, const NodePtr &expert_tokens_count_or_cumsum_flag,
                                      const NodePtr &expert_tokens_before_capacity_flag) {
  return NativeFunc::MoeInitRoutingV2(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                      expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag);
}

NodePtr FuncBuilder::MoeGatingTopKSoftmax(const NodePtr &x, const NodePtr &finished, const NodePtr &k) {
  return NativeFunc::MoeGatingTopKSoftmax(x, finished, k);
}

NodePtr FuncBuilder::MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2,
                                        const NodePtr &bias, const NodePtr &scales, const NodePtr &expanded_row_idx,
                                        const NodePtr &expanded_expert_idx) {
  return NativeFunc::MoeFinalizeRouting(expanded_x, x1, x2, bias, scales, expanded_row_idx, expanded_expert_idx);
}

NodePtr FuncBuilder::MoeInitRoutingQuantV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                                           const NodePtr &expert_capacity, const NodePtr &expert_num,
                                           const NodePtr &drop_pad_mode,
                                           const NodePtr &expert_tokens_count_or_cumsum_flag,
                                           const NodePtr &expert_tokens_before_capacity_flag, const NodePtr &quant_mode,
                                           const NodePtr &scale, const NodePtr &offset) {
  return NativeFunc::MoeInitRoutingQuantV2(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                           expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                                           quant_mode, scale, offset);
}

NodePtr FuncBuilder::DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) {
  return NativeFunc::DynamicQuantExt(x, smooth_scales);
}

NodePtr FuncBuilder::KVCacheScatterUpdate(const NodePtr &var, const NodePtr &indices, const NodePtr &updates,
                                          const NodePtr &axis, const NodePtr &reduce) {
  return NativeFunc::KVCacheScatterUpdate(var, indices, updates, axis, reduce);
}

NodePtr FuncBuilder::FuncDropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &training,
                                    const NodePtr &inplace, const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::FuncDropoutExt(input, p, training, inplace, seed, offset);
}

NodePtr FuncBuilder::GmmBackward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight,
                                 const NodePtr &group_list, const NodePtr &group_list_type) {
  return NativeFunc::GmmBackward(grad, x, weight, group_list, group_list_type);
}

NodePtr FuncBuilder::PixelShuffle(const NodePtr &input, const NodePtr &upscale_factor) {
  return NativeFunc::PixelShuffle(input, upscale_factor);
}

NodePtr FuncBuilder::GmmV2BackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                                         const NodePtr &group_list_type) {
  return NativeFunc::GmmV2BackwardFusion(grad, weight, group_list, group_list_type);
}

NodePtr FuncBuilder::AnyExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::AnyExt(input, dim, keepdim);
}

NodePtr FuncBuilder::GmmV2Backward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight,
                                   const NodePtr &group_list, const NodePtr &group_list_type) {
  return NativeFunc::GmmV2Backward(grad, x, weight, group_list, group_list_type);
}

NodePtr FuncBuilder::FuncMaxPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                                   const NodePtr &padding, const NodePtr &dilation, const NodePtr &ceil_mode,
                                   const NodePtr &return_indices) {
  return NativeFunc::FuncMaxPool2D(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices);
}

NodePtr FuncBuilder::MoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices,
                                       const NodePtr &probs, const NodePtr &padded_mode, const NodePtr &restore_shape) {
  return NativeFunc::MoeTokenUnpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

NodePtr FuncBuilder::Any(const NodePtr &input) { return NativeFunc::Any(input); }

NodePtr FuncBuilder::InplaceExponential(const NodePtr &input, const NodePtr &lambd, const NodePtr &seed,
                                        const NodePtr &offset) {
  return NativeFunc::InplaceExponential(input, lambd, seed, offset);
}

NodePtr FuncBuilder::Dropout2dExt(const NodePtr &input, const NodePtr &p, const NodePtr &training,
                                  const NodePtr &inplace, const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::Dropout2dExt(input, p, training, inplace, seed, offset);
}

NodePtr FuncBuilder::GmmBackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                                       const NodePtr &group_list_type) {
  return NativeFunc::GmmBackwardFusion(grad, weight, group_list, group_list_type);
}

NodePtr FuncBuilder::Gmm(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
                         const NodePtr &group_type, const NodePtr &group_list_type) {
  return NativeFunc::Gmm(x, weight, bias, group_list, group_type, group_list_type);
}

NodePtr FuncBuilder::EinsumExt(const NodePtr &equation, const NodePtr &operands) {
  return NativeFunc::EinsumExt(equation, operands);
}

NodePtr FuncBuilder::GmmV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
                           const NodePtr &group_type, const NodePtr &group_list_type) {
  return NativeFunc::GmmV2(x, weight, bias, group_list, group_type, group_list_type);
}
}  // namespace mindspore::pynative::autograd
