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

#include <map>
#include <set>
#include <string>
#include <memory>

#include "ops/grad/cholesky_grad.h"
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
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CholeskyGradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_shapetrack =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto x_dims = x_shape.size();
  auto grad_dims = grad_shape.size();
  size_t kIndex1 = x_dims - 1;
  size_t kIndex2 = x_dims - 2;
  if (x_dims != grad_dims) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x and grad should have same dims, but got" << x_dims
                             << " and " << grad_dims << ".";
  }
  if (x_shape[kIndex1] != x_shape[kIndex2]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x's last two dim size should be same, but got"
                             << x_shape[kIndex2] << " and " << x_shape[kIndex1] << ".";
  }
  if (grad_shape[kIndex1] != grad_shape[kIndex2]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", grad's last two dim size should be same, but got"
                             << grad_shape[kIndex2] << " and " << grad_shape[kIndex1] << ".";
  }
  if (grad_shape[kIndex1] != x_shape[kIndex1]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << "x and grad's last dim size should be same, but got"
                             << x_shape[kIndex1] << " and " << grad_shape[kIndex1] << ".";
  }
  return std::make_shared<abstract::Shape>(x_shapetrack);
}
TypePtr CholeskyGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto grad_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("x", x_type);
  (void)args.emplace("grad", grad_type);
  return CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(CholeskyGrad, BaseOperator);
AbstractBasePtr CholeskyGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = CholeskyGradInferType(primitive, input_args);
  auto infer_shape = CholeskyGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCholeskyGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CholeskyGrad, prim::kPrimCholeskyGrad, AGCholeskyGradInfer, false);
}  // namespace ops
}  // namespace mindspore
