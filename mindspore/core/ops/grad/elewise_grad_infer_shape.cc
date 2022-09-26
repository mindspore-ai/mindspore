/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/elewise_grad_infer_shape.h"
#include <vector>
#include <memory>
#include "utils/check_convert_utils.h"

namespace mindspore {
abstract::ShapePtr ElewiseGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto dout = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto x_shape_ptr = x->shape();
  auto dout_shape_ptr = dout->shape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  MS_EXCEPTION_IF_NULL(dout_shape_ptr);
  auto x_shape = x_shape_ptr->shape();
  auto dout_shape = dout_shape_ptr->shape();
  if (IsDynamicRank(x_shape) || IsDynamicRank(dout_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  if (x_shape.size() != dout_shape.size()) {
    MS_EXCEPTION(RuntimeError) << "Rank of x(" << x_shape.size() << ") and dout(" << dout_shape.size()
                               << ") not equal, primitive name: " << prim_name << ".";
  }
  ShapeVector output_shape(x_shape.size());
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] != dout_shape[i]) {
      if (x_shape[i] == abstract::Shape::kShapeDimAny || dout_shape[i] == abstract::Shape::kShapeDimAny) {
        output_shape[i] = abstract::Shape::kShapeDimAny;
      } else {
        MS_EXCEPTION(RuntimeError) << "The " << i << "th dim of x(" << x_shape[i] << ") and dout(" << dout_shape[i]
                                   << ") not equal.";
      }
    } else {
      output_shape[i] = x_shape[i];
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}
}  // namespace mindspore
