/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/shape.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  // infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_prim = primitive->cast<PrimShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_prim);
  auto op_name = shape_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), op_name);
  // infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<abstract::AbstractTensor>(x_type, in_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Shape, prim::kPrimShape, ShapeInfer);
REGISTER_PRIMITIVE_C(kNameShape, Shape);
}  // namespace ops
}  // namespace mindspore
