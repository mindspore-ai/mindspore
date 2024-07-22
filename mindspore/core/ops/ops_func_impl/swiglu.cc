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

#include "ops/ops_func_impl/swiglu.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr SwigluInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  constexpr size_t kSplitNum = 2;
  int64_t dim = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue()).value();
  if (IsDynamicRank(x_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  const size_t x_rank = x_shape.size();
  if (dim < 0) {
    dim += x_rank;
  }
  x_shape[dim] = x_shape[dim] / kSplitNum;
  return std::make_shared<abstract::TensorShape>(x_shape);
}

TypePtr SwigluInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return input_args[kInputIndex0]->GetType();
}
}  // namespace

BaseShapePtr SwigluFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return SwigluInferShape(primitive, input_args);
}

TypePtr SwigluFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return SwigluInferType(primitive, input_args);
}

}  // namespace ops
}  // namespace mindspore
