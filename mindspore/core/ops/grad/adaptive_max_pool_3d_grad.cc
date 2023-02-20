/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/adaptive_max_pool_3d_grad.h"

#include <map>
#include <set>

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
abstract::ShapePtr AdaptiveMaxPool3DGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 1);
  auto argmax_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 2);
  if (!x_shape_ptr->IsDynamic() && !argmax_shape_ptr->IsDynamic() && !argmax_shape_ptr->IsDynamic()) {
    auto input_grad_shape = input_grad_shape_ptr->shape();
    auto x_shape = x_shape_ptr->shape();
    auto argmax_shape = argmax_shape_ptr->shape();
    const int64_t input_grad_dims = SizeToLong(input_grad_shape.size());
    const int64_t x_dims = SizeToLong(x_shape.size());
    const int64_t argmax_dim = SizeToLong(argmax_shape.size());
    CheckAndConvertUtils::CheckInRange("dim of input_grad", input_grad_dims, kIncludeBoth, {4, 5}, prim_name);
    CheckAndConvertUtils::CheckInRange("dim of x", x_dims, kIncludeBoth, {4, 5}, prim_name);
    CheckAndConvertUtils::CheckInRange("dim of argmax", argmax_dim, kIncludeBoth, {4, 5}, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dim of input_grad", input_grad_dims, kEqual, x_dims, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dim of argmax", argmax_dim, kEqual, x_dims, prim_name);
    if (input_grad_shape != argmax_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', input_grad shape must be same with argmax shape, but got "
                               << input_grad_shape_ptr->ToString() << " and " << argmax_shape_ptr->ToString() << ".";
    }
  }
  return x_shape_ptr;
}

TypePtr AdaptiveMaxPool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_grad_dtype = input_args[0]->BuildType();
  auto x_dtype = input_args[1]->BuildType();
  auto argmax_dtype = input_args[2]->BuildType();
  const std::set<TypePtr> common_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                                kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> argmax_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad", input_grad_dtype, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", argmax_dtype, argmax_valid_types, prim_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_grad", input_grad_dtype);
  (void)types.emplace("x", x_dtype);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, primitive->name());
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool3DGrad, BaseOperator);
AbstractBasePtr AdaptiveMaxPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveMaxPool3DGradInferType(primitive, input_args);
  auto shapes = AdaptiveMaxPool3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGAdaptiveMaxPool3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveMaxPool3DGrad, prim::kPrimAdaptiveMaxPool3DGrad, AGAdaptiveMaxPool3DGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
