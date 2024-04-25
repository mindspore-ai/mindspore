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

#include "ops/ops_func_impl/dropout_do_mask_ext.h"
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
bool IsKeepProbValid(const AbstractBasePtr &keep_prob_abs) {
  auto keep_prob_opt = GetArrayValue<T>(keep_prob_abs);
  if (!keep_prob_opt.has_value()) {
    return false;
  }
  auto keep_prob = keep_prob_opt.value()[kIndex0];
  if (keep_prob > static_cast<T>(1.0) || keep_prob < static_cast<T>(0.0)) {
    MS_EXCEPTION(ValueError) << "For 'DropoutDoMaskExt', the 'keep_prob(1-p)' must be in range [0, 1], but got "
                             << keep_prob << ".";
  }
  return true;
}
}  // namespace

int32_t DropoutDoMaskExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args[kIndex2]->GetType()->object_type() == kObjectTypeTensorType) {
    // p will be replaced with keep_prob after some pass.
    auto keep_prob_shape = input_args[kIndex2]->GetShape()->GetShapeVector();
    MS_CHECK_VALUE(keep_prob_shape.empty(),
                   "For 'DropoutDoMaskExt', the dim of 'keep_prob' must be 0(scalar), but got " +
                     std::to_string(keep_prob_shape.size()) + ".");
    auto keep_prob_dtype = input_args[kIndex2]->GetType()->cast<TensorTypePtr>()->element()->type_id();
    static std::set<TypeId> valid_dtype_set = {kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeBFloat16};
    MS_CHECK_VALUE(valid_dtype_set.find(keep_prob_dtype) != valid_dtype_set.end(),
                   "For 'DropoutDoMaskExt', the keep_prob type must be in [Float32, Float16, BFloat16], but got " +
                     TypeIdToString(keep_prob_dtype));
    if (MS_UNLIKELY(keep_prob_dtype == kNumberTypeFloat16 && !IsKeepProbValid<float16>(input_args[kIndex2]))) {
      return OP_CHECK_RETRY;
    } else if (MS_UNLIKELY(keep_prob_dtype == kNumberTypeBFloat16 && !IsKeepProbValid<bfloat16>(input_args[kIndex2]))) {
      return OP_CHECK_RETRY;
    } else if (MS_UNLIKELY(keep_prob_dtype == kNumberTypeFloat32 && !IsKeepProbValid<float>(input_args[kIndex2]))) {
      return OP_CHECK_RETRY;
    }
  } else {
    auto p_opt = GetScalarValue<float>(input_args[kIndex2]->GetValue());
    if (MS_UNLIKELY(!p_opt.has_value())) {
      return OP_CHECK_RETRY;
    }
    MS_CHECK_VALUE(
      p_opt.value() >= static_cast<float>(0.0) && p_opt.value() <= static_cast<float>(1.0),
      "For 'DropoutDoMaskExt', the 'p' must be in range [0, 1], but got " + std::to_string(p_opt.value()) + ".");
  }

  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto mask_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(input_shape) || IsDynamic(mask_shape))) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(mask_shape.size() == 1, "For 'DropoutDoMaskExt', the 'mask' must be 1-D, but got " +
                                           std::to_string(mask_shape.size()) + "-D.");
  auto input_size = std::accumulate(input_shape.cbegin(), input_shape.cend(), 1, std::multiplies<int64_t>());
  auto mask_size = mask_shape[kIndex0] * 8;
  if (input_size > mask_size) {
    MS_EXCEPTION(ValueError)
      << "For 'DropoutDoMaskExt', the input 'mask' must be less than or equal to match input, but got 'input' shape: "
      << ShapeVectorToString(input_shape) << ", 'mask' shape: " << ShapeVectorToString(mask_shape) << ".";
  }

  return OP_CHECK_SUCCESS;
}

BaseShapePtr DropoutDoMaskExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr DropoutDoMaskExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
