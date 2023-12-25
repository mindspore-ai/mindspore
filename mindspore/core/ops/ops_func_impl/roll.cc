/**
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

#include "ops/ops_func_impl/roll.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
int32_t RollFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &axis_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(!axis_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  const auto &shift_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  if (MS_UNLIKELY(!shift_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  const auto &axis_array = axis_opt.value();
  const auto &shift_array = shift_opt.value();
  if (axis_array.size() != shift_array.size() || axis_array.size() == 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', 'axis' and 'shift' must be not empty and have same size.";
  }
  return OP_CHECK_SUCCESS;
}

BaseShapePtr RollFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetShape()->Clone();
}

TypePtr RollFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
