/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/generate_eod_mask_v2.h"

#include <vector>
#include <set>
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GenerateEodMaskV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr GenerateEodMaskV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  const auto &op_name = primitive->name();
  auto input_type = input_args[kIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, {kBFloat16, kFloat16, kFloat32}, op_name);

  const std::set<TypePtr> valid_types{kInt64};
  for (size_t i = kIndex1; i < kIndex5; i++) {
    const auto &type = input_args[i]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid(tensor_arg_names_[i], type, valid_types, op_name);
  }

  return input_type;
}

int32_t GenerateEodMaskV2FuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  for (size_t i = kIndex2; i < kIndex5; i++) {
    CheckTensorScalarRank(primitive, input_args[i], tensor_arg_names_[i]);
  }

  auto start_opt = GetScalarValue<int64_t>(input_args[kIndex5]->GetValue());
  if (MS_UNLIKELY(!start_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto start = start_opt.value();
  MS_CHECK_VALUE(start >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg("start", start, kGreaterEqual, 0, primitive));

  auto steps_opt = GetArrayValue<int64_t>(input_args[kIndex6]);
  if (MS_UNLIKELY(!steps_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto steps = steps_opt.value();
  MS_CHECK_VALUE(steps.size() >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg("number of steps", steps.size(),
                                                                                kGreaterEqual, 1, primitive));
  for (size_t i = 0; i < steps.size(); ++i) {
    if (MS_UNLIKELY(steps.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(steps[i] >= 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps[i], kGreaterEqual, 0, primitive));
  }

  auto bit_pos_opt = GetScalarValue<int64_t>(input_args[kIndex10]->GetValue());
  if (MS_UNLIKELY(!bit_pos_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto bit_pos = bit_pos_opt.value();
  auto input_type = input_args[kIndex0]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type);
  auto bit_size = SizeToLong(abstract::TypeIdSize(input_type->element()->type_id()) * kIndex8);
  MS_CHECK_VALUE(bit_pos >= 0 && bit_pos < bit_size, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                       "bit pos", bit_pos, kIncludeLeft, {0, bit_size}, primitive));

  return OP_CHECK_SUCCESS;
}

}  // namespace ops
}  // namespace mindspore
