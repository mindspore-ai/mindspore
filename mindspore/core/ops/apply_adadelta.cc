/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "ops/apply_adadelta.h"

#include <set>
#include <map>
#include <utility>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdadeltaInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto accum_shape = input_args[kInputIndex1]->BuildShape();
  auto accum_update_shape = input_args[kInputIndex2]->BuildShape();
  auto grad_shape = input_args[kInputIndex6]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto accum_shape_ptr = accum_shape->cast<abstract::ShapePtr>();
  auto accum_update_shape_ptr = accum_update_shape->cast<abstract::ShapePtr>();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  // var and accum must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !accum_shape_ptr->IsDynamic()) {
    if (*var_shape != *accum_shape) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', 'var' and 'accum' must have the same shape when is not dynamic. But got 'var' shape: "
        << var_shape->ToString() << ", 'accum' shape: " << accum_shape->ToString() << ".";
    }
  }
  // var and accum update must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !accum_update_shape_ptr->IsDynamic()) {
    if (*var_shape != *accum_update_shape) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', 'var' and 'accum_update' must have the same shape when is not dynamic. But got 'var' shape: "
        << var_shape->ToString() << ", 'accum_update' shape: " << accum_update_shape->ToString() << ".";
    }
  }
  // var and grad must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !grad_shape_ptr->IsDynamic()) {
    if (*var_shape != *grad_shape) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', 'var' and 'grad' must have the same shape when is not dynamic. But got 'var' shape: "
        << var_shape->ToString() << ", 'grad' shape: " << grad_shape->ToString() << ".";
    }
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, accum_shape, accum_update_shape});
}

TuplePtr ApplyAdadeltaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto accum_update_type = input_args[kInputIndex2]->BuildType();
  auto lr_type = input_args[kInputIndex3]->BuildType();
  auto rho_type = input_args[kInputIndex4]->BuildType();
  auto epsilon_type = input_args[kInputIndex5]->BuildType();
  auto grad_type = input_args[kInputIndex6]->BuildType();

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("accum_type", accum_type));
  (void)args.insert(std::make_pair("accum_update_type", accum_update_type));
  (void)args.insert(std::make_pair("grad_type", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.insert(std::make_pair("lr_type", lr_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);

  std::map<std::string, TypePtr> args_rho;
  (void)args_rho.insert(std::make_pair("rho_type", rho_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_rho, valid_types, prim_name);

  std::map<std::string, TypePtr> args_epsilon;
  (void)args_epsilon.insert(std::make_pair("epsilon_type", epsilon_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_epsilon, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type, accum_update_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(ApplyAdadelta, BaseOperator);
AbstractBasePtr ApplyAdadeltaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = ApplyAdadeltaInferType(primitive, input_args);
  auto infer_shape = ApplyAdadeltaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGApplyAdadeltaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdadeltaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdadeltaInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdadeltaInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyAdadelta, prim::kPrimApplyAdadelta, AGApplyAdadeltaInfer, false);
}  // namespace ops
}  // namespace mindspore
