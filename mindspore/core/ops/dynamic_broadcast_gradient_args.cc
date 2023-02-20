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

#include <string>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
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
  ShapeVector max_shape;
  // DynamicBroadcastGradientArgs is a compute depend op
  if (x_shape0 >= 0 && y_shape0 >= 0) {
    max_shape = {x_shape0 > y_shape0 ? x_shape0 : y_shape0};
    // Currently, if the max_shape is 0, there may be some problems
    max_shape[0] = max_shape[0] != 0 ? max_shape[0] : 1;
  }

  auto out_shape = std::make_shared<abstract::Shape>(shape, max_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicBroadcastGradientArgs, BaseOperator);

class MIND_API DynamicBroadcastGradientArgsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Infer(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto types = std::vector<TypePtr>{kInt64, kInt64};
    auto output_type = std::make_shared<Tuple>(types);
    return output_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicBroadcastGradientArgs, prim::kPrimDynamicBroadcastGradientArgs,
                                 DynamicBroadcastGradientArgsInfer, false);
}  // namespace ops
}  // namespace mindspore
