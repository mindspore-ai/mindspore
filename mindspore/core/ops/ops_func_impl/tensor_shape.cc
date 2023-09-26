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

#include "ops/ops_func_impl/tensor_shape.h"
#include <memory>
#include <vector>
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ir/dtype/number.h"
#include "utils/shape_utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore::ops {
BaseShapePtr TensorShapeFuncImpl::InferShape(const PrimitivePtr &,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto base_shape = input_args[kInputIndex0]->GetShape();
  auto shape = base_shape->GetShapeVector();
  if (IsDynamicRank(shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeDimAny});
  }
  return std::make_shared<abstract::TensorShape>(ShapeVector{SizeToLong(shape.size())});
}

TypePtr TensorShapeFuncImpl::InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) const {
  return std::make_shared<TensorType>(kInt64);
}
}  // namespace mindspore::ops
