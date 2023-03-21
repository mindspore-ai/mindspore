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

#include "ops/apply_gradient_descent.h"

#include <set>
#include <utility>
#include <map>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ApplyGradientDescentInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape)[kShape])) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (IsDynamicRank(alpha_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto delta_shape = input_args[kInputIndex2]->BuildShape();
  if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(delta_shape)[kShape])) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  // var and delta must have the same shape when is not dynamic
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto delta_shape_ptr = delta_shape->cast<abstract::ShapePtr>();
  if (!var_shape_ptr->IsDynamic() && !delta_shape_ptr->IsDynamic()) {
    if (*var_shape != *delta_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', evaluator arg 'delta' must have the same shape as 'var'. But got 'delta' shape: "
                               << delta_shape->ToString() << ", 'var' shape: " << var_shape->ToString() << ".";
    }
  }
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    batch_rank = GetValue<int64_t>(primitive->GetAttr(kBatchRank));
  }
  // alpha must be a scalar [Number, Tensor]
  const int64_t kShapeSize = 1;
  auto alpha_shape_rank = SizeToLong(alpha_shape.size());
  if (batch_rank > 0) {
    // when batch dimension exists, the rank of `alpha` must equal to batch_rank.
    (void)CheckAndConvertUtils::CheckInteger("alpha's rank'", alpha_shape_rank, kEqual, batch_rank, prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("alpha's rank'", alpha_shape_rank, kLessEqual, kShapeSize, prim_name);
    if (alpha_shape_rank == 1) {
      (void)CheckAndConvertUtils::CheckInteger("alpha_shape[0]", alpha_shape[0], kEqual, kShapeSize, primitive->name());
    }
  }
  MS_EXCEPTION_IF_NULL(var_shape_ptr);
  return var_shape_ptr;
}

TypePtr ApplyGradientDescentInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto alpha_type = input_args[kInputIndex1]->BuildType();
  auto delta_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt8,   kUInt8,   kInt16,     kUInt16,    kInt32,
                                         kUInt32,  kInt64,   kUInt64, kFloat64, kComplex64, kComplex128};
  // delta must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("delta_type", delta_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // alpha must be a scalar type
  std::map<std::string, TypePtr> args_alpha;
  (void)args_alpha.insert(std::make_pair("alpha_type", alpha_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_alpha, valid_types, prim_name);
  return var_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ApplyGradientDescent, BaseOperator);
AbstractBasePtr ApplyGradientDescentInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = ApplyGradientDescentInferType(primitive, input_args);
  auto infer_shape = ApplyGradientDescentInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGApplyGradientDescentInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyGradientDescentInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyGradientDescentInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyGradientDescentInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyGradientDescent, prim::kPrimApplyGradientDescent, AGApplyGradientDescentInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
