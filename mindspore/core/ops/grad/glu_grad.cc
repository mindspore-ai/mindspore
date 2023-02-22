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
#include "ops/grad/glu_grad.h"

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
#include "abstract/utils.h"
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
abstract::ShapePtr GluGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto grad = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(grad);
  auto x = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto x_shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape_element);
  auto grad_shape_element = grad->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(grad_shape_element);
  if (!x_shape_element->IsDynamic() && !grad_shape_element->IsDynamic()) {
    auto x_shape = x_shape_element->shape();
    auto grad_shape = grad_shape_element->shape();
    auto x_rank = SizeToLong(x_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("rank of x", x_rank, kGreaterEqual, 1, prim_name);
    auto axis_value = GetValue<int64_t>(primitive->GetAttr("axis"));
    CheckAndConvertUtils::CheckInRange("axis", axis_value, kIncludeLeft, {-x_rank, x_rank}, prim_name);
    auto axis = axis_value;
    if (axis < 0) {
      axis += x_rank;
    }
    const int64_t kEvenNum = 2;
    if (x_shape[axis] % kEvenNum != 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x.shape[" << axis_value << "] must be even, but got "
                               << x_shape[axis] << ".";
    }

    auto expected_grad_shape = x_shape;
    expected_grad_shape[axis] /= kEvenNum;
    if (grad_shape != expected_grad_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', x.shape must be euqal to grad.shape except for grad.shape[axis]=x.shape[axis]"
                                  "/2,  but got axis="
                               << axis_value << ", x.shape=" << x_shape_element->ToString()
                               << " and grad.shape=" << grad_shape_element->ToString() << ".";
    }
  }

  return x_shape_element;
}

TypePtr GluGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto dy = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, y, dy);
  auto x_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> input_types = {kFloat64, kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, input_types, primitive->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GluGrad, BaseOperator);
AbstractBasePtr GluGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto shape = GluGradInferShape(primitive, input_args);
  auto type = GluGradInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGGluGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GluGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GluGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GluGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GluGrad, prim::kPrimGluGrad, AGGluGradInfer, false);
}  // namespace ops
}  // namespace mindspore
