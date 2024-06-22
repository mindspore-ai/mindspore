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

#include "ops/ops_func_impl/trace.h"

#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"

namespace mindspore::ops {
BaseShapePtr TraceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &shape = base_shape->GetShapeVector();

  const size_t kTraceInputRank = 2;
  if (!IsDynamic(shape) && shape.size() != kTraceInputRank) {
    MS_LOG(EXCEPTION) << "For Primitive[Trace], the rank of the input must be 2, but got " << shape.size() << "!";
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtr TraceFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kInt8,  kInt16,  kInt32,  kInt64,  kFloat16,   kFloat32,   kFloat64,
                                         kUInt8, kUInt16, kUInt32, kUInt64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, primitive->name());
  return input_type;
}
}  // namespace mindspore::ops
