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

#include "pynative/forward/pyboost/do_pyboost_cast.h"
#include "pynative/backward/grad_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/auto_generate/cast.h"
#include "include/utils/pynative/common_utils.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"

namespace mindspore {
namespace pynative {
ValuePtr PyBoostCastOperation::DoAutoCast(const PyboostOpRunInfoPtr &op_run_info,
                                          const std::pair<TypeId, bool> &dst_type, size_t index,
                                          const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtr dst_value = CastUtils::ScalarToDstDtypeValue(v, dst_type);
  if (dst_value != nullptr) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString() << " cast to value: " << dst_value->ToString();
    return dst_value;
  }
  if (!v->isa<tensor::Tensor>()) {
    return v;
  }
  return DoAutoCast(op_run_info, dst_type, index, v->cast<tensor::TensorPtr>());
}

tensor::TensorPtr PyBoostCastOperation::DoAutoCast(const PyboostOpRunInfoPtr &op_run_info,
                                                   const std::pair<TypeId, bool> &dst_type, size_t index,
                                                   const tensor::TensorPtr &t) const {
  if (op_run_info->source_type[index] != ops::OP_DTYPE::DT_BEGIN) {
    if (!EnableDispatch()) {
      MS_LOG(DEBUG) << "[Dispatch]Try cast Source tensor: " << t->ToString();
      auto dst_tensor = CastUtils::TensorToDstDtypeValue(t, dst_type.first);
      MS_LOG(DEBUG) << "[Dispatch]Cast to dst tensor: " << dst_tensor->ToString() << " without dispatching cast op";
      return dst_tensor;
    }
  }
  auto type_id64 = std::make_shared<Int64Imm>(static_cast<int64_t>(dst_type.first));
  kernel::pyboost::RequireGradGuard require_grad(op_run_info->requires_grad);
  return kernel::pyboost::cast(t, type_id64);
}

ValuePtr PyBoostCastOperation::SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                                         size_t index) const {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return SetTensorMixPrecisionCast(op_run_info, v->cast<tensor::TensorPtr>(), index);
  }
  return v;
}

tensor::TensorPtr PyBoostCastOperation::SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info,
                                                                  const tensor::TensorPtr &t, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(t);
  auto dst_dtype = kFloat16;
  if (op_run_info->mix_precision_type == nullptr) {
    if (op_run_info->mix_type == kFP32) {
      dst_dtype = kFloat32;
    } else if (op_run_info->mix_type == kBF16) {
      dst_dtype = kBFloat16;
    }
  } else {
    dst_dtype = op_run_info->mix_precision_type;
  }
  auto source_dtype = t->Dtype();
  if (source_dtype != nullptr && (IsSubType(source_dtype, kFloat) || IsSubType(source_dtype, kBFloat)) &&
      *source_dtype != *dst_dtype) {
    MS_LOG(DEBUG) << "MixPrecision cast for " << op_run_info->op_prim->name() << " " << index
                  << "th input, and to type " << dst_dtype->ToString();
    auto cast_t = DoAutoCast(op_run_info, std::make_pair(dst_dtype->type_id(), true), index, t);
    return cast_t;
  }
  return t;
}

std::optional<tensor::TensorPtr> PyBoostCastOperation::SetTensorMixPrecisionCast(
  const PyboostOpRunInfoPtr &op_run_info, const std::optional<tensor::TensorPtr> &t, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!t.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(SetTensorMixPrecisionCast(op_run_info, t.value(), index));
}

ValueTuplePtr PyBoostCastOperation::SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info,
                                                              const ValueTuplePtr &v_tuple, size_t index) const {
  return std::make_shared<ValueTuple>(SetSeqMixPrecisionCast(op_run_info, v_tuple, index));
}

ValueListPtr PyBoostCastOperation::SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info,
                                                             const ValueListPtr &v_list, size_t index) const {
  return std::make_shared<ValueList>(SetSeqMixPrecisionCast(op_run_info, v_list, index));
}

ValuePtrList PyBoostCastOperation::SetSeqMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info,
                                                          const ValueSequencePtr &v_seq, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v_seq);
  size_t tuple_size = v_seq->size();
  const auto &value_tuple = v_seq->value();
  ValuePtrList result(tuple_size, nullptr);
  for (size_t i = 0; i < tuple_size; i++) {
    if (value_tuple[i]->isa<tensor::MetaTensor>()) {
      MS_LOG(DEBUG) << "Call cast for " << i << "th input";
      result[i] = SetTensorMixPrecisionCast(op_run_info, value_tuple[i], index);
    } else if (value_tuple[i]->isa<ValueTuple>()) {
      result[i] = SetTensorMixPrecisionCast(op_run_info, value_tuple[i]->cast<ValueTuplePtr>(), index);
    } else if (value_tuple[i]->isa<ValueList>()) {
      result[i] = SetTensorMixPrecisionCast(op_run_info, value_tuple[i]->cast<ValueListPtr>(), index);
    } else {
      result[i] = value_tuple[i];
    }
  }
  return result;
}
}  // namespace pynative
}  // namespace mindspore
