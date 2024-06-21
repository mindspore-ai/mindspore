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

#include <memory>
#include <set>
#include <string>
#include "ops/ops_func_impl/normal_float_tensor.h"
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr NormalFloatTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  if (!CheckAndConvertUtils::IsTensor(input_args[kInputIndex0]) &&
      CheckAndConvertUtils::IsTensor(input_args[kInputIndex1])) {
    auto std_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
    if (IsDynamicRank(std_shape)) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
    }
    return std::make_shared<abstract::TensorShape>(std_shape);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', mean must be a Tensor, std must be a Float, but got: "
                            << input_args[kInputIndex0]->ToString() << " and, " << input_args[kInputIndex1]->ToString()
                            << ".";
  }
}

TypePtr NormalFloatTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto x_type = input_args[1]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types_with_complex_and_bool, prim_name);
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
