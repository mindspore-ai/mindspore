/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/forward/do_pyboost_cast.h"
#include "pipeline/pynative/pynative_utils.h"
#include "kernel/pyboost/ops/cast.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace pynative {
tensor::TensorPtr PyBoostCastOperation::DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const TypeId &type_id,
                                                   size_t index, const tensor::TensorPtr &t) const {
  if (op_run_info->source_type[index] != ops::OP_DTYPE::DT_BEGIN) {
    MS_LOG(DEBUG) << "Try cast Source tensor: " << t->ToString();
    auto dst_tensor = TensorToDstDtypeValue(t, type_id);
    MS_LOG(DEBUG) << "Cast to dst tensor: " << dst_tensor->ToString() << " without dispatching cast op";
    return dst_tensor;
  }
  const auto &cast_run_info = std::make_shared<FrontendOpRunInfo>();
  auto cast_prim = GetPrimByTypeId(type_id);
  // Use pyboost op call
  cast_run_info->base_op_run_info.device_target =
    PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->GetCurrentDeviceTarget(cast_prim);
  auto cast_op = CREATE_PYBOOST_OP(Cast, cast_run_info->base_op_run_info.device_target);
  cast_op->set_primitive(cast_prim);
  auto new_type = GetDstTypeValue(type_id);
  (void)cast_op->Call(t, new_type->cast<TypePtr>());
  PyNativeAlgo::PyBoost::UpdateOpRunInfo(cast_op, cast_run_info->op_grad_info->input_value, cast_run_info);
  if (op_run_info->requires_grad) {
    constexpr auto input_size = 2;
    cast_run_info->input_size = input_size;
    cast_run_info->base_op_run_info.op_name = kCast;
    (void)cast_run_info->op_grad_info->input_value.emplace_back(t);
    (void)cast_run_info->op_grad_info->input_value.emplace_back(new_type);
    cast_run_info->op_grad_info->op_prim = cast_prim;
    cast_run_info->requires_grad = true;
    cast_op->DoGrad();
  }
  return cast_run_info->real_out->cast<tensor::TensorPtr>();
}

tensor::TensorPtr PyBoostCastOperation::SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info,
                                                                  const tensor::TensorPtr &t, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(t);
  const auto &signature = op_run_info->signatures;
  if (t->is_parameter()) {
    // If parameter write(not kRWRead), no need cast
    if (signature[index].rw != SignatureEnumRW::kRWRead) {
      return t;
    }
  }

  if (op_run_info->mix_type != kNotSet) {
    auto dst_dtype = kFloat16;
    if (op_run_info->mix_type == kFP32) {
      dst_dtype = kFloat32;
    } else if (op_run_info->mix_type == kBF16) {
      dst_dtype = kBFloat16;
    }

    auto source_dtype = t->Dtype();
    if (source_dtype != nullptr && (IsSubType(source_dtype, kFloat) || IsSubType(source_dtype, kBFloat)) &&
        *source_dtype != *dst_dtype) {
      MS_LOG(DEBUG) << "MixPrecision cast for " << op_run_info->base_op_run_info.op_name << " " << index
                    << "th input, and to type " << dst_dtype->ToString();
      auto cast_t = DoAutoCast(op_run_info, dst_dtype->type_id(), index, t);
      return cast_t;
    }
  }
  return t;
}

std::optional<tensor::TensorPtr> PyBoostCastOperation::SetTensorMixPrecisionCast(
  const FrontendOpRunInfoPtr &op_run_info, const std::optional<tensor::TensorPtr> &t, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!t.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(SetTensorMixPrecisionCast(op_run_info, t.value(), index));
}

ValuePtr PyBoostCastOperation::SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info,
                                                         const ValueSequencePtr &v_seq, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v_seq);

  size_t tuple_size = v_seq->size();
  const auto &value_tuple = v_seq->value();
  ValuePtrList result(tuple_size, nullptr);
  for (size_t i = 0; i < tuple_size; i++) {
    if (value_tuple[i]->isa<tensor::MetaTensor>()) {
      MS_LOG(DEBUG) << "Call cast for item " << i;
      result[i] = SetTensorMixPrecisionCast(op_run_info, value_tuple[i], index);
    } else if (value_tuple[i]->isa<ValueSequence>()) {
      result[i] = SetTensorMixPrecisionCast(op_run_info, value_tuple[i]->cast<ValueSequencePtr>(), index);
    } else {
      result[i] = value_tuple[i];
    }
  }

  if (v_seq->isa<ValueList>()) {
    return std::make_shared<ValueList>(result);
  } else {
    return std::make_shared<ValueTuple>(result);
  }
}

}  // namespace pynative
}  // namespace mindspore
