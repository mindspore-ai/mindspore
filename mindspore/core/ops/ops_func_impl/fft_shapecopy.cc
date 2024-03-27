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

#include "ops/ops_func_impl/fft_shapecopy.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTShapeCopyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto dout_shape_ptr = input_args[kIndex0]->GetShape();
  auto dout_shape = dout_shape_ptr->GetShapeVector();
  if (IsDynamicRank(dout_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto x_shape = dout_shape;
  auto shape_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (shape_opt.has_value()) {
    std::vector<int64_t> shape = shape_opt.value().ToVector();
    for (size_t i = 0; i < shape.size(); i++) {
      x_shape[i] = shape[i];
    }
  }

  return std::make_shared<abstract::TensorShape>(x_shape);
}

TypePtr FFTShapeCopyFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
