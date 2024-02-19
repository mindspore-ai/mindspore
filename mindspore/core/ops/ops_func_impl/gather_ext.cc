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
#include "ops/ops_func_impl/gather_ext.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GatherExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex2]->GetShape()->Clone();
}

TypePtr GatherExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto index_type = input_args[kInputIndex2]->GetType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", index_type, valid_types, primitive->name());
  return input_args[kInputIndex0]->GetType()->Clone();
}

int32_t GatherExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(input_shape)) {
    return OP_CHECK_RETRY;
  }

  auto dim = input_args[kInputIndex1]->GetValue();
  MS_EXCEPTION_IF_NULL(dim);
  auto dim_value = GetScalarValue<int64_t>(dim);
  if (!dim_value.has_value()) {
    return OP_CHECK_RETRY;
  }

  auto upper_bound = static_cast<int64_t>(input_shape.size());
  auto low_bound_include = -upper_bound;
  upper_bound = upper_bound == 0 ? 1 : upper_bound;
  MS_CHECK_VALUE(
    low_bound_include <= dim_value.value() && dim_value.value() < upper_bound,
    CheckAndConvertUtils::FormatCommMsg("For Primitive[", primitive->name(), "], the dim must be in range [",
                                        low_bound_include, ", ", upper_bound, "), but got: ", dim_value.value(), "."));

  auto index_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  if (IsDynamicRank(index_shape)) {
    return OP_CHECK_RETRY;
  }

  input_shape = input_shape.empty() ? ShapeVector({1}) : input_shape;
  index_shape = index_shape.empty() ? ShapeVector({1}) : index_shape;
  MS_CHECK_VALUE(
    index_shape.size() == input_shape.size(),
    CheckAndConvertUtils::FormatCommMsg("For Primitive[", primitive->name(),
                                        "], the rank of input and index must be equal, but got input_shape[",
                                        input_shape, "] and index_shape[", index_shape, "]."));

  for (size_t i = 0; i < index_shape.size(); i++) {
    if (dim_value.value() != static_cast<int64_t>(i)) {
      if (input_shape[i] == abstract::Shape::kShapeDimAny || index_shape[i] == abstract::Shape::kShapeDimAny) {
        return OP_CHECK_RETRY;
      }

      MS_CHECK_VALUE(index_shape[i] <= input_shape[i],
                     CheckAndConvertUtils::FormatCommMsg("For Primitive[", primitive->name(),
                                                         "], index_shape[d] must not be great than "
                                                         "input_shape[d] while d != dim, but got input_shape[",
                                                         input_shape, "] and index_shape[", index_shape, "]."));
    }
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
