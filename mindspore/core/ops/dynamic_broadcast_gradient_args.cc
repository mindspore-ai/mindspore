/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/dynamic_broadcast_gradient_args.h"

#include <set>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CheckInputsAndGetShape(const AbstractBasePtr &input_arg, const string &prim_name) {
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<abstract::AbstractTensor>()) {
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_arg->BuildShape())[kShape];
    auto input_size = input_shape.size();
    if (input_size != 1) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input shape must be 1-D, but got: " << input_size << "-D.";
    }
    return input_shape[0];
  } else if (input_arg->isa<abstract::AbstractTuple>()) {
    auto x_shape = dyn_cast<abstract::AbstractTuple>(input_arg);
    auto x_shape_data = x_shape->elements();
    return SizeToLong(x_shape_data.size());
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input type must be a tuple or Tensor.";
  }
}

abstract::TupleShapePtr Infer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto x_shape0 = CheckInputsAndGetShape(input_args[0], prim_name);
  auto y_shape0 = CheckInputsAndGetShape(input_args[1], prim_name);

  ShapeVector shape{abstract::Shape::kShapeDimAny};
  if (x_shape0 < 0 && y_shape0 < 0) {
    auto out_shape = std::make_shared<abstract::Shape>(shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }

  auto out_shape = std::make_shared<abstract::Shape>(shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicBroadcastGradientArgs, BaseOperator);
AbstractBasePtr DynamicBroadcastGradientArgsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto types = std::vector<TypePtr>{kInt64, kInt64};
  auto output_type = std::make_shared<Tuple>(types);
  return abstract::MakeAbstract(Infer(primitive, input_args), output_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(DynamicBroadcastGradientArgs, prim::kPrimDynamicBroadcastGradientArgs,
                             DynamicBroadcastGradientArgsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
