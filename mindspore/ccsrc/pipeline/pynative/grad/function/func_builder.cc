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
#include "pipeline/pynative/grad/function/auto_generate/pyboost_native_grad_functions.h"

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
  if (value_ptr->isa<tensor::BaseTensor>()) {
    auto tensor = value_ptr->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    // In pynative mode, need data sync before get tensor value, otherwise the tensor value may be undefined.
    tensor->data_sync();
    return CheckAndConvertUtils::CheckTensorIntValue("value", value_ptr, "GetIntList");
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

FuncBuilder::FuncBuilder(const std::string &name, std::string device_target, const expander::ExpanderInferPtr &infer)
    : BpropBuilder(name, infer), device_target_(device_target) {
  pass_forward_ = std::make_shared<bprop_pass::FuncPassForward>(this, std::move(device_target));
  NativeFunc::set_device_target(device_target_);
}

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

NodePtr FuncBuilder::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) {
  return NativeFunc::MatMul(a, b, Value(transpose_a), Value(transpose_b));
}

NodePtr FuncBuilder::MatMulExt(const NodePtr &a, const NodePtr &b) {
  auto [input, mat] = UnifyDtype2(a, b);
  return NativeFunc::MatMulExt(input, mat);
}

NodePtr FuncBuilder::Add(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  return NativeFunc::Add(input, other);
}
NodePtr FuncBuilder::Sub(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  return NativeFunc::Sub(input, other);
}
NodePtr FuncBuilder::Mul(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  return NativeFunc::Mul(input, other);
}
NodePtr FuncBuilder::Div(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  return NativeFunc::Div(input, other);
}

NodePtr FuncBuilder::Pow(const NodePtr &lhs, const NodePtr &rhs) {
  auto [input, exponent] = UnifyDtype2(lhs, rhs);
  return NativeFunc::Pow(input, exponent);
}

NodePtr FuncBuilder::Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto abs = lhs->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    auto [input, other] = UnifyDtype2(lhs, rhs);
    auto node = NativeFunc::Equal(input, other);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  } else if (abs->isa<abstract::AbstractScalar>()) {
    return ScalarEq(lhs, rhs, dst_type);
  }
  MS_LOG(EXCEPTION) << "'Equal' only support [Tensor] or [Scalar] input, but got: " << abs->ToString();
}

NodePtr FuncBuilder::NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  auto node = NativeFunc::NotEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  auto node = NativeFunc::GreaterEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  auto node = NativeFunc::Greater(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::LessEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  auto node = NativeFunc::LessEqual(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Less(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
  auto [input, other] = UnifyDtype2(lhs, rhs);
  auto node = NativeFunc::Less(input, other);
  return dst_type == nullptr ? node : Cast(node, dst_type);
}

NodePtr FuncBuilder::Concat(const NodePtr &tensors, const NodePtr &axis) {
  tensors->SetValue(FillZeros(tensors->Value(), tensors->abstract()));
  return NativeFunc::Concat(tensors, axis);
}

NodePtr FuncBuilder::Abs(const NodePtr &input) { return NativeFunc::Abs(input); }

NodePtr FuncBuilder::AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v,
                           const NodePtr &gradient, const NodePtr &step, const NodePtr &lr, const NodePtr &beta1,
                           const NodePtr &beta2, const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad,
                           const NodePtr &maximize) {
  return NativeFunc::AdamW(var, m, v, max_v, gradient, step, lr, beta1, beta2, decay, eps, amsgrad, maximize);
}

NodePtr FuncBuilder::AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::AddExt(input, other, alpha);
}

NodePtr FuncBuilder::AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                                    const NodePtr &epsilon, const NodePtr &additionalOut) {
  return NativeFunc::AddLayerNormV2(x1, x2, gamma, beta, epsilon, additionalOut);
}

NodePtr FuncBuilder::Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                           const NodePtr &alpha) {
  return NativeFunc::Addmm(input, mat1, mat2, beta, alpha);
}

NodePtr FuncBuilder::Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) {
  return NativeFunc::Arange(start, end, step, dtype);
}

NodePtr FuncBuilder::ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
  return NativeFunc::ArgMaxExt(input, dim, keepdim);
}

NodePtr FuncBuilder::ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ArgMaxWithValue(input, axis, keep_dims);
}

NodePtr FuncBuilder::ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ArgMinWithValue(input, axis, keep_dims);
}

NodePtr FuncBuilder::Atan2Ext(const NodePtr &input, const NodePtr &other) { return NativeFunc::Atan2Ext(input, other); }

NodePtr FuncBuilder::AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                   const NodePtr &count_include_pad, const NodePtr &divisor_override) {
  return NativeFunc::AvgPool2DGrad(grad, image, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                   divisor_override);
}

NodePtr FuncBuilder::AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                               const NodePtr &divisor_override) {
  return NativeFunc::AvgPool2D(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

NodePtr FuncBuilder::BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                                 const NodePtr &transpose_b) {
  return NativeFunc::BatchMatMul(x, y, transpose_a, transpose_b);
}

NodePtr FuncBuilder::BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                  const NodePtr &running_mean, const NodePtr &runnning_var, const NodePtr &training,
                                  const NodePtr &momentum, const NodePtr &epsilon) {
  return NativeFunc::BatchNormExt(input, weight, bias, running_mean, runnning_var, training, momentum, epsilon);
}

NodePtr FuncBuilder::BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                      const NodePtr &running_mean, const NodePtr &running_var,
                                      const NodePtr &saved_mean, const NodePtr &saved_rstd, const NodePtr &training,
                                      const NodePtr &eps) {
  return NativeFunc::BatchNormGradExt(dout, input, weight, running_mean, running_var, saved_mean, saved_rstd, training,
                                      eps);
}

NodePtr FuncBuilder::BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                            const NodePtr &weight, const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropyGrad(input, target, grad_output, weight, reduction);
}

NodePtr FuncBuilder::BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                        const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropy(input, target, weight, reduction);
}

NodePtr FuncBuilder::BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input,
                                                          const NodePtr &target, const NodePtr &weight,
                                                          const NodePtr &posWeight, const NodePtr &reduction) {
  return NativeFunc::BinaryCrossEntropyWithLogitsBackward(grad_output, input, target, weight, posWeight, reduction);
}

NodePtr FuncBuilder::BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                       const NodePtr &posWeight, const NodePtr &reduction) {
  return NativeFunc::BCEWithLogitsLoss(input, target, weight, posWeight, reduction);
}

NodePtr FuncBuilder::BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) {
  return NativeFunc::BatchMatMulExt(input, mat2);
}

NodePtr FuncBuilder::BroadcastTo(const NodePtr &input, const NodePtr &shape) {
  return NativeFunc::BroadcastTo(input, shape);
}

NodePtr FuncBuilder::Ceil(const NodePtr &input) { return NativeFunc::Ceil(input); }

NodePtr FuncBuilder::Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
  return NativeFunc::Chunk(input, chunks, dim);
}

NodePtr FuncBuilder::ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::ClampScalar(input, min, max);
}

NodePtr FuncBuilder::ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
  return NativeFunc::ClampTensor(input, min, max);
}

NodePtr FuncBuilder::Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                               const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Col2ImExt(input, output_size, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                                const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Col2ImGrad(input, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) {
  return NativeFunc::ConstantPadND(input, padding, value);
}

NodePtr FuncBuilder::Contiguous(const NodePtr &input) { return NativeFunc::Contiguous(input); }

NodePtr FuncBuilder::ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                     const NodePtr &bias, const NodePtr &stride, const NodePtr &padding,
                                     const NodePtr &dilation, const NodePtr &transposed, const NodePtr &output_padding,
                                     const NodePtr &groups, const NodePtr &output_mask) {
  return NativeFunc::ConvolutionGrad(dout, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                     groups, output_mask);
}

NodePtr FuncBuilder::Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                 const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                 const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups) {
  return NativeFunc::Convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

NodePtr FuncBuilder::Copy(const NodePtr &input) { return NativeFunc::Copy(input); }

NodePtr FuncBuilder::Cos(const NodePtr &input) { return NativeFunc::Cos(input); }

NodePtr FuncBuilder::CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
  return NativeFunc::CumsumExt(input, dim, dtype);
}

NodePtr FuncBuilder::Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) {
  return NativeFunc::Dense(input, weight, bias);
}

NodePtr FuncBuilder::DivMod(const NodePtr &x, const NodePtr &y, const NodePtr &rounding_mode) {
  return NativeFunc::DivMod(x, y, rounding_mode);
}

NodePtr FuncBuilder::Dot(const NodePtr &input, const NodePtr &other) { return NativeFunc::Dot(input, other); }

NodePtr FuncBuilder::DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
  return NativeFunc::DropoutDoMaskExt(input, mask, p);
}

NodePtr FuncBuilder::DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) {
  return NativeFunc::DropoutExt(input, p, seed, offset);
}

NodePtr FuncBuilder::DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed,
                                       const NodePtr &offset, const NodePtr &dtype) {
  return NativeFunc::DropoutGenMaskExt(shape, p, seed, offset, dtype);
}

NodePtr FuncBuilder::DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
  return NativeFunc::DropoutGradExt(input, mask, p);
}

NodePtr FuncBuilder::EluExt(const NodePtr &input, const NodePtr &alpha) { return NativeFunc::EluExt(input, alpha); }

NodePtr FuncBuilder::EluGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &alpha) {
  return NativeFunc::EluGradExt(dout, x, alpha);
}

NodePtr FuncBuilder::EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                            const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) {
  return NativeFunc::EmbeddingDenseBackward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

NodePtr FuncBuilder::Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx,
                               const NodePtr &max_norm, const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) {
  return NativeFunc::Embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq);
}

NodePtr FuncBuilder::Erf(const NodePtr &input) { return NativeFunc::Erf(input); }

NodePtr FuncBuilder::Erfinv(const NodePtr &input) { return NativeFunc::Erfinv(input); }

NodePtr FuncBuilder::Exp(const NodePtr &input) { return NativeFunc::Exp(input); }

NodePtr FuncBuilder::Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) {
  return NativeFunc::Eye(n, m, dtype);
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

NodePtr FuncBuilder::FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
  return NativeFunc::FillScalar(size, fill_value, dtype);
}

NodePtr FuncBuilder::FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
  return NativeFunc::FillTensor(size, fill_value, dtype);
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

NodePtr FuncBuilder::FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) {
  return NativeFunc::FlattenExt(input, start_dim, end_dim);
}

NodePtr FuncBuilder::Floor(const NodePtr &input) { return NativeFunc::Floor(input); }

NodePtr FuncBuilder::GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) {
  return NativeFunc::GatherDGradV2(x, dim, index, dout);
}

NodePtr FuncBuilder::GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) {
  return NativeFunc::GatherD(x, dim, index);
}

NodePtr FuncBuilder::GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) {
  return NativeFunc::GeLUGrad(dy, x, y);
}

NodePtr FuncBuilder::GeLU(const NodePtr &input) { return NativeFunc::GeLU(input); }

NodePtr FuncBuilder::Generator(const NodePtr &cmd, const NodePtr &inputs) { return NativeFunc::Generator(cmd, inputs); }

NodePtr FuncBuilder::GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                       const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                       const NodePtr &align_corners) {
  return NativeFunc::GridSampler2DGrad(grad, input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                   const NodePtr &padding_mode, const NodePtr &align_corners) {
  return NativeFunc::GridSampler2D(input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                       const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                       const NodePtr &align_corners) {
  return NativeFunc::GridSampler3DGrad(grad, input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                   const NodePtr &padding_mode, const NodePtr &align_corners) {
  return NativeFunc::GridSampler3D(input_x, grid, interpolation_mode, padding_mode, align_corners);
}

NodePtr FuncBuilder::GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                                   const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                                   const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) {
  return NativeFunc::GroupNormGrad(dy, x, mean, rstd, gamma_opt, num_groups, dx_is_require, dgamma_is_require,
                                   dbeta_is_require);
}

NodePtr FuncBuilder::GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight,
                               const NodePtr &bias, const NodePtr &eps) {
  return NativeFunc::GroupNorm(input, num_groups, weight, bias, eps);
}

NodePtr FuncBuilder::Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                               const NodePtr &padding, const NodePtr &stride) {
  return NativeFunc::Im2ColExt(input, kernel_size, dilation, padding, stride);
}

NodePtr FuncBuilder::IndexAddExt(const NodePtr &input, const NodePtr &index, const NodePtr &source, const NodePtr &axis,
                                 const NodePtr &alpha) {
  return NativeFunc::IndexAddExt(input, index, source, axis, alpha);
}

NodePtr FuncBuilder::IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
  return NativeFunc::IndexSelect(input, dim, index);
}

NodePtr FuncBuilder::IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                             const NodePtr &equal_nan) {
  return NativeFunc::IsClose(input, other, rtol, atol, equal_nan);
}

NodePtr FuncBuilder::IsFinite(const NodePtr &x) { return NativeFunc::IsFinite(x); }

NodePtr FuncBuilder::LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                                  const NodePtr &bias, const NodePtr &eps) {
  return NativeFunc::LayerNormExt(input, normalized_shape, weight, bias, eps);
}

NodePtr FuncBuilder::LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape,
                                      const NodePtr &mean, const NodePtr &variance, const NodePtr &gamma,
                                      const NodePtr &beta) {
  return NativeFunc::LayerNormGradExt(dy, x, normalized_shape, mean, variance, gamma, beta);
}

NodePtr FuncBuilder::LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) {
  return NativeFunc::LeakyReLUExt(input, negative_slope);
}

NodePtr FuncBuilder::LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                                      const NodePtr &is_result) {
  return NativeFunc::LeakyReLUGradExt(dy, input, negative_slope, is_result);
}

NodePtr FuncBuilder::LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) {
  return NativeFunc::LinSpaceExt(start, end, steps, dtype);
}

NodePtr FuncBuilder::LogicalAnd(const NodePtr &x, const NodePtr &y) { return NativeFunc::LogicalAnd(x, y); }

NodePtr FuncBuilder::LogicalNot(const NodePtr &input) { return NativeFunc::LogicalNot(input); }

NodePtr FuncBuilder::LogicalOr(const NodePtr &x, const NodePtr &y) { return NativeFunc::LogicalOr(x, y); }

NodePtr FuncBuilder::MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) {
  return NativeFunc::MaskedFill(input_x, mask, value);
}

NodePtr FuncBuilder::MatrixInverseExt(const NodePtr &input) { return NativeFunc::MatrixInverseExt(input); }

NodePtr FuncBuilder::Max(const NodePtr &input) { return NativeFunc::Max(input); }

NodePtr FuncBuilder::MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                            const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                            const NodePtr &dilation, const NodePtr &ceil_mode,
                                            const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolGradWithIndices(x, grad, argmax, kernel_size, strides, pads, dilation, ceil_mode,
                                            argmax_type);
}

NodePtr FuncBuilder::MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask,
                                         const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                         const NodePtr &dilation, const NodePtr &ceil_mode,
                                         const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolGradWithMask(x, grad, mask, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                        const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                        const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolWithIndices(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                     const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                     const NodePtr &argmax_type) {
  return NativeFunc::MaxPoolWithMask(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
}

NodePtr FuncBuilder::Maximum(const NodePtr &input, const NodePtr &other) { return NativeFunc::Maximum(input, other); }

NodePtr FuncBuilder::MeanExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims,
                             const NodePtr &dtype) {
  return NativeFunc::MeanExt(input, axis, keep_dims, dtype);
}

NodePtr FuncBuilder::Min(const NodePtr &input) { return NativeFunc::Min(input); }

NodePtr FuncBuilder::Minimum(const NodePtr &input, const NodePtr &other) { return NativeFunc::Minimum(input, other); }

NodePtr FuncBuilder::Mv(const NodePtr &input, const NodePtr &vec) { return NativeFunc::Mv(input, vec); }

NodePtr FuncBuilder::Neg(const NodePtr &input) { return NativeFunc::Neg(input); }

NodePtr FuncBuilder::NonZeroExt(const NodePtr &input) { return NativeFunc::NonZeroExt(input); }

NodePtr FuncBuilder::NonZero(const NodePtr &input) { return NativeFunc::NonZero(input); }

NodePtr FuncBuilder::Norm(const NodePtr &input_x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
                          const NodePtr &dtype) {
  return NativeFunc::Norm(input_x, ord, dim, keepdim, dtype);
}

NodePtr FuncBuilder::NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                                      const NodePtr &offset) {
  return NativeFunc::NormalFloatFloat(mean, std, size, seed, offset);
}

NodePtr FuncBuilder::NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                       const NodePtr &offset) {
  return NativeFunc::NormalFloatTensor(mean, std, seed, offset);
}

NodePtr FuncBuilder::NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                       const NodePtr &offset) {
  return NativeFunc::NormalTensorFloat(mean, std, seed, offset);
}

NodePtr FuncBuilder::NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                        const NodePtr &offset) {
  return NativeFunc::NormalTensorTensor(mean, std, seed, offset);
}

NodePtr FuncBuilder::OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                               const NodePtr &off_value, const NodePtr &axis) {
  return NativeFunc::OneHotExt(tensor, num_classes, on_value, off_value, axis);
}

NodePtr FuncBuilder::OnesLikeExt(const NodePtr &input, const NodePtr &dtype) {
  return NativeFunc::OnesLikeExt(input, dtype);
}

NodePtr FuncBuilder::Ones(const NodePtr &shape, const NodePtr &dtype) { return NativeFunc::Ones(shape, dtype); }

NodePtr FuncBuilder::ProdExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims,
                             const NodePtr &dtype) {
  return NativeFunc::ProdExt(input, axis, keep_dims, dtype);
}

NodePtr FuncBuilder::RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) {
  return NativeFunc::RandExt(shape, seed, offset, dtype);
}

NodePtr FuncBuilder::RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset,
                                 const NodePtr &dtype) {
  return NativeFunc::RandLikeExt(tensor, seed, offset, dtype);
}

NodePtr FuncBuilder::Reciprocal(const NodePtr &x) { return NativeFunc::Reciprocal(x); }

NodePtr FuncBuilder::ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceAll(input, axis, keep_dims);
}

NodePtr FuncBuilder::ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
  return NativeFunc::ReduceAny(x, axis, keep_dims);
}

NodePtr FuncBuilder::ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad1DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReflectionPad1D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad1D(input, padding);
}

NodePtr FuncBuilder::ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad2DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReflectionPad2D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad2D(input, padding);
}

NodePtr FuncBuilder::ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad3DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReflectionPad3D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReflectionPad3D(input, padding);
}

NodePtr FuncBuilder::ReluGrad(const NodePtr &y_backprop, const NodePtr &x) {
  return NativeFunc::ReluGrad(y_backprop, x);
}

NodePtr FuncBuilder::ReLU(const NodePtr &input) { return NativeFunc::ReLU(input); }

NodePtr FuncBuilder::RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) {
  return NativeFunc::RepeatInterleaveGrad(input, repeats, dim);
}

NodePtr FuncBuilder::RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                         const NodePtr &output_size) {
  return NativeFunc::RepeatInterleaveInt(input, repeats, dim, output_size);
}

NodePtr FuncBuilder::RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                            const NodePtr &output_size) {
  return NativeFunc::RepeatInterleaveTensor(input, repeats, dim, output_size);
}

NodePtr FuncBuilder::ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad1DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReplicationPad1D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad1D(input, padding);
}

NodePtr FuncBuilder::ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad2DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReplicationPad2D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad2D(input, padding);
}

NodePtr FuncBuilder::ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad3DGrad(grad_output, input, padding);
}

NodePtr FuncBuilder::ReplicationPad3D(const NodePtr &input, const NodePtr &padding) {
  return NativeFunc::ReplicationPad3D(input, padding);
}

NodePtr FuncBuilder::ReverseV2(const NodePtr &input, const NodePtr &axis) { return NativeFunc::ReverseV2(input, axis); }

NodePtr FuncBuilder::RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) {
  return NativeFunc::RmsNormGrad(dy, x, rstd, gamma);
}

NodePtr FuncBuilder::RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) {
  return NativeFunc::RmsNorm(x, gamma, epsilon);
}

NodePtr FuncBuilder::Rsqrt(const NodePtr &input) { return NativeFunc::Rsqrt(input); }

NodePtr FuncBuilder::ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) {
  return NativeFunc::ScatterAddExt(input, dim, index, src);
}

NodePtr FuncBuilder::Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                             const NodePtr &reduce) {
  return NativeFunc::Scatter(input, dim, index, src, reduce);
}

NodePtr FuncBuilder::SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                                  const NodePtr &dtype, const NodePtr &right) {
  return NativeFunc::SearchSorted(sorted_sequence, values, sorter, dtype, right);
}

NodePtr FuncBuilder::Select(const NodePtr &condition, const NodePtr &input, const NodePtr &other) {
  return NativeFunc::Select(condition, input, other);
}

NodePtr FuncBuilder::SigmoidGrad(const NodePtr &y, const NodePtr &dy) { return NativeFunc::SigmoidGrad(y, dy); }

NodePtr FuncBuilder::Sigmoid(const NodePtr &input) { return NativeFunc::Sigmoid(input); }

NodePtr FuncBuilder::Sign(const NodePtr &input) { return NativeFunc::Sign(input); }

NodePtr FuncBuilder::SiLUGrad(const NodePtr &dout, const NodePtr &x) { return NativeFunc::SiLUGrad(dout, x); }

NodePtr FuncBuilder::SiLU(const NodePtr &input) { return NativeFunc::SiLU(input); }

NodePtr FuncBuilder::Sin(const NodePtr &input) { return NativeFunc::Sin(input); }

NodePtr FuncBuilder::SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                              const NodePtr &step) {
  return NativeFunc::SliceExt(input, dim, start, end, step);
}

NodePtr FuncBuilder::SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) {
  return NativeFunc::SoftmaxBackward(dout, out, dim);
}

NodePtr FuncBuilder::Softmax(const NodePtr &input, const NodePtr &axis) { return NativeFunc::Softmax(input, axis); }

NodePtr FuncBuilder::SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) {
  return NativeFunc::SoftplusExt(input, beta, threshold);
}

NodePtr FuncBuilder::SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                                     const NodePtr &threshold) {
  return NativeFunc::SoftplusGradExt(dout, x, beta, threshold);
}

NodePtr FuncBuilder::SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending,
                             const NodePtr &stable) {
  return NativeFunc::SortExt(input, dim, descending, stable);
}

NodePtr FuncBuilder::SplitTensor(const NodePtr &input_x, const NodePtr &split_int, const NodePtr &axis) {
  return NativeFunc::SplitTensor(input_x, split_int, axis);
}

NodePtr FuncBuilder::SplitWithSize(const NodePtr &input_x, const NodePtr &split_sections, const NodePtr &axis) {
  return NativeFunc::SplitWithSize(input_x, split_sections, axis);
}

NodePtr FuncBuilder::Sqrt(const NodePtr &x) { return NativeFunc::Sqrt(x); }

NodePtr FuncBuilder::Square(const NodePtr &input) { return NativeFunc::Square(input); }

NodePtr FuncBuilder::StackExt(const NodePtr &tensors, const NodePtr &dim) { return NativeFunc::StackExt(tensors, dim); }

NodePtr FuncBuilder::SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
  return NativeFunc::SubExt(input, other, alpha);
}

NodePtr FuncBuilder::SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
  return NativeFunc::SumExt(input, dim, keepdim, dtype);
}

NodePtr FuncBuilder::TanhGrad(const NodePtr &y, const NodePtr &dy) { return NativeFunc::TanhGrad(y, dy); }

NodePtr FuncBuilder::Tanh(const NodePtr &input) { return NativeFunc::Tanh(input); }

NodePtr FuncBuilder::Tile(const NodePtr &input, const NodePtr &dims) { return NativeFunc::Tile(input, dims); }

NodePtr FuncBuilder::TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                             const NodePtr &sorted) {
  return NativeFunc::TopkExt(input, k, dim, largest, sorted);
}

NodePtr FuncBuilder::Triu(const NodePtr &input, const NodePtr &diagonal) { return NativeFunc::Triu(input, diagonal); }

NodePtr FuncBuilder::UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                                const NodePtr &offset) {
  return NativeFunc::UniformExt(tensor, a, b, seed, offset);
}

NodePtr FuncBuilder::Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                             const NodePtr &return_counts) {
  return NativeFunc::Unique2(input, sorted, return_inverse, return_counts);
}

NodePtr FuncBuilder::UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                               const NodePtr &dim) {
  return NativeFunc::UniqueDim(input, sorted, return_inverse, dim);
}

NodePtr FuncBuilder::UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                            const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleBilinear2DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                        const NodePtr &align_corners) {
  return NativeFunc::UpsampleBilinear2D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                          const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleLinear1DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                      const NodePtr &align_corners) {
  return NativeFunc::UpsampleLinear1D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest1DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest1D(x, output_size, scales);
}

NodePtr FuncBuilder::UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest2DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest2D(x, output_size, scales);
}

NodePtr FuncBuilder::UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                           const NodePtr &scales) {
  return NativeFunc::UpsampleNearest3DGrad(dy, input_size, output_size, scales);
}

NodePtr FuncBuilder::UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
  return NativeFunc::UpsampleNearest3D(x, output_size, scales);
}

NodePtr FuncBuilder::UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                             const NodePtr &scales, const NodePtr &align_corners) {
  return NativeFunc::UpsampleTrilinear3DGrad(dy, input_size, output_size, scales, align_corners);
}

NodePtr FuncBuilder::UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                         const NodePtr &align_corners) {
  return NativeFunc::UpsampleTrilinear3D(x, output_size, scales, align_corners);
}

NodePtr FuncBuilder::ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) {
  return NativeFunc::ZerosLikeExt(input, dtype);
}

NodePtr FuncBuilder::Zeros(const NodePtr &size, const NodePtr &dtype) { return NativeFunc::Zeros(size, dtype); }

NodePtr FuncBuilder::DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) {
  return NativeFunc::DynamicQuantExt(x, smooth_scales);
}

NodePtr FuncBuilder::GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                   const NodePtr &offset, const NodePtr &antiquant_scale,
                                   const NodePtr &antiquant_offset, const NodePtr &group_list,
                                   const NodePtr &split_item, const NodePtr &group_type) {
  return NativeFunc::GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                   split_item, group_type);
}

NodePtr FuncBuilder::MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2,
                                        const NodePtr &bias, const NodePtr &scales, const NodePtr &expanded_row_idx,
                                        const NodePtr &expanded_expert_idx) {
  return NativeFunc::MoeFinalizeRouting(expanded_x, x1, x2, bias, scales, expanded_row_idx, expanded_expert_idx);
}

NodePtr FuncBuilder::QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                                      const NodePtr &bias, const NodePtr &transpose_x1, const NodePtr &transpose_x2,
                                      const NodePtr &dtype) {
  return NativeFunc::QuantBatchMatmul(x1, x2, scale, offset, bias, transpose_x1, transpose_x2, dtype);
}

NodePtr FuncBuilder::QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                             const NodePtr &rounding_mode, const NodePtr &dst_type) {
  return NativeFunc::QuantV2(x, scale, offset, sqrt_mode, rounding_mode, dst_type);
}

NodePtr FuncBuilder::WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                            const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                            const NodePtr &quant_offset, const NodePtr &bias,
                                            const NodePtr &transpose_x, const NodePtr &transpose_weight,
                                            const NodePtr &antiquant_group_size) {
  return NativeFunc::WeightQuantBatchMatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset,
                                            bias, transpose_x, transpose_weight, antiquant_group_size);
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
