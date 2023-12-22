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

#include "ops/ops_func_impl/cosh.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CoshFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto base_shape = input_args[kInputIndex0]->GetShape();
  const auto x_shape = base_shape->GetShapeVector();
  const size_t max_dim = 9;
  MS_CHECK_VALUE(x_shape.size() < max_dim, CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", x_shape.size(),
                                                                                       kLessThan, max_dim, primitive));
  return std::make_shared<abstract::TensorShape>(x_shape);
}

TypePtr CoshFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
