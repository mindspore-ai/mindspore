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

#include "ops/ops_func_impl/eye.h"
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"

namespace mindspore {
namespace ops {
BaseShapePtr EyeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  int64_t n = abstract::Shape::kShapeDimAny;
  int64_t m = abstract::Shape::kShapeDimAny;
  auto n_ptr = GetScalarValue<int64_t>(input_args[0]->GetValue());
  auto m_ptr = GetScalarValue<int64_t>(input_args[1]->GetValue());

  if (n_ptr.has_value()) {
    n = n_ptr.value();
    MS_CHECK_VALUE(n >= 0, primitive->name() + " error: n value can not be negative.");
  }
  if (m_ptr.has_value()) {
    m = m_ptr.value();
    MS_CHECK_VALUE(m >= 0, primitive->name() + " error: m value can not be negative.");
  }

  return std::make_shared<abstract::TensorShape>(ShapeVector{n, m});
}

TypePtr EyeFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  return std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_ptr.value())));
}
}  // namespace ops
}  // namespace mindspore
