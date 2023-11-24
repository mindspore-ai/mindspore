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

#include "ops/ops_func_impl/resize_nearest_neighbor.h"

#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
BaseShapePtr ResizeNearestNeighborFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  ShapeVector output_shape;
  const int64_t size_size = 2;
  // support dynamic rank
  if (IsDynamicRank(x_shape)) {
    output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    output_shape.push_back(abstract::TensorShape::kShapeDimAny);
  } else {
    const int64_t shape_size = 4;
    MS_CHECK_VALUE(
      x_shape.size() == shape_size,
      CheckAndConvertUtils::FormatCommMsg("For '", primitive->name(), "', the dimension of input_x should equal to ",
                                          shape_size, " but got ", x_shape.size()));
    output_shape.insert(output_shape.end(), x_shape.begin(), x_shape.begin() + size_size);
  }

  auto size_ptr = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (!size_ptr.has_value()) {
    output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(output_shape);
  }
  auto size_array = size_ptr.value();
  MS_CHECK_VALUE(
    size_array.size() == size_size,
    CheckAndConvertUtils::FormatCommMsg("For '", primitive->name(), "', the dimension of size should equal to ",
                                        size_size, " but got ", size_array.size()));

  for (size_t i = 0; i < size_array.size(); ++i) {
    if (MS_UNLIKELY(size_array.IsValueUnknown(i))) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
      continue;
    }
    MS_CHECK_VALUE(size_array[i] >= 0, CheckAndConvertUtils::FormatCommMsg(
                                         "For '", primitive->name(),
                                         "', the value of size should greater or equal to 0, but got ", size_array[i]));
    output_shape.push_back(size_array[i]);
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr ResizeNearestNeighborFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
