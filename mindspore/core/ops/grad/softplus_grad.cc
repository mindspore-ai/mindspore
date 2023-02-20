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
#include "ops/grad/softplus_grad.h"

#include <string>
#include <set>
#include <vector>
#include <memory>

#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SoftplusGradInfershape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(output_shape);
  auto x_shape_ptr = x_shape->cast<abstract::ShapePtr>();
  auto output_shape_ptr = output_shape->cast<abstract::ShapePtr>();
  if (!x_shape_ptr->IsDynamic() && !output_shape_ptr->IsDynamic()) {
    if (*x_shape != *output_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', evaluator arg 'x' and 'output' must have the same shape, but got 'x' shape: "
                               << x_shape->ToString() << ", 'output' shape: " << output_shape->ToString() << ".";
    }
  }
  auto shape_element = x_shape_ptr;
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SoftplusGradInfertype(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto output = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, x, output);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SoftplusGrad, BaseOperator);
class MIND_API SoftplusGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftplusGradInfershape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftplusGradInfertype(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SoftplusGrad, prim::kPrimSoftplusGrad, SoftplusGradInfer, false);
}  // namespace ops
}  // namespace mindspore
