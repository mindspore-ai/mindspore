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

#include "ops/ops_func_impl/index_add_ext.h"
#include <memory>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr IndexAddExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr IndexAddExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

int32_t IndexAddExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(input_shape))) {
    return OP_CHECK_RETRY;
  }

  auto source_shape = input_args[kIndex2]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(source_shape))) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(input_shape.size() == source_shape.size(),
                 "For 'IndexAddExt', the input.ndim must equal to source.ndim, but got input.ndim " +
                   std::to_string(input_shape.size()) + ", and source.ndim " + std::to_string(source_shape.size()));
  int64_t input_rank = SizeToLong(input_shape.size());

  auto axis_opt = GetScalarValue<int64_t>(input_args[kIndex3]->GetValue());
  if (MS_UNLIKELY(!axis_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  if (MS_UNLIKELY(axis_opt.value() >= input_rank || axis_opt.value() < -input_rank)) {
    MS_EXCEPTION(ValueError) << "For 'IndexAddExt', the axis must be in '[" << -input_rank << ", " << input_rank
                             << ")', but got " << axis_opt.value() << ".";
  }
  auto axis = axis_opt.value() < 0 ? axis_opt.value() + input_rank : axis_opt.value();
  for (int64_t idx = 0; idx < input_rank; ++idx) {
    if (idx == axis) {
      continue;
    }
    MS_CHECK_VALUE(input_shape[idx] == source_shape[idx],
                   "For 'IndexAddExt', the input's shape must be same as source except the 'axis' dim, but got input[" +
                     ShapeVectorToString(input_shape) + "] and source[" + ShapeVectorToString(source_shape) + "].");
  }

  auto index_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(index_shape))) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(index_shape[kIndex0] == source_shape[axis],
                 "For 'IndexAddExt', the size of index must be equal to source[axis], but got source's shape [" +
                   ShapeVectorToString(source_shape) + "], size of index " + std::to_string(index_shape[kIndex0]) +
                   ", axis " + std::to_string(axis_opt.value()) + ".");

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
