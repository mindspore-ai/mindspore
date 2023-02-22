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

#include "ops/grad/adaptive_max_pool2d_grad.h"

#include <set>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>

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
#include "mindapi/base/shape_vector.h"
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
constexpr size_t inputArgLen = 3;
constexpr int64_t kDynamicRankVal = -2;

abstract::ShapePtr AdaptiveMaxPool2DGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto y_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];

  std::vector<ShapeVector> all_shapes = {y_grad_shape, x_shape, argmax_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  if (is_dynamic_rank) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{kDynamicRankVal});
  }

  const int64_t y_grad_dims = SizeToLong(y_grad_shape.size());
  const int64_t x_dims = SizeToLong(x_shape.size());
  const int64_t argmax_dims = SizeToLong(argmax_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("y_grad_dims", y_grad_dims, kEqual, x_dims, kNameAdaptiveMaxPool2DGrad);
  (void)CheckAndConvertUtils::CheckInteger("argmax_dims", argmax_dims, kEqual, x_dims, kNameAdaptiveMaxPool2DGrad);

  CheckAndConvertUtils::CheckInRange("y_grad_dim", y_grad_dims, kIncludeBoth, {3, 4}, kNameAdaptiveMaxPool2DGrad);
  CheckAndConvertUtils::CheckInRange("x_dim", x_dims, kIncludeBoth, {3, 4}, kNameAdaptiveMaxPool2DGrad);
  CheckAndConvertUtils::CheckInRange("argmax_dim", argmax_dims, kIncludeBoth, {3, 4}, kNameAdaptiveMaxPool2DGrad);

  auto is_dynamic = IsDynamic(y_grad_shape) || IsDynamic(argmax_shape);
  if (!is_dynamic && y_grad_shape != argmax_shape) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the shape of 'y_grad' should be consistent with the shape of 'argmax'.";
  }

  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr AdaptiveMaxPool2DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto y_grad_dtype = input_args[0]->BuildType();
  auto argmax_dtype = input_args[2]->BuildType();

  const std::set<TypePtr> floating_data_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> argmax_valid_types = {kInt64};

  std::map<std::string, TypePtr> args;
  (void)args.emplace("y_grad", input_args[0]->BuildType());
  (void)args.emplace("x", input_args[1]->BuildType());

  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, floating_data_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax_dtype", argmax_dtype, argmax_valid_types,
                                                   kNameAdaptiveMaxPool2DGrad);
  return y_grad_dtype;
}
}  // namespace

AbstractBasePtr AdaptiveMaxPool2DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveMaxPool2DGradInferType(primitive, input_args);
  auto shapes = AdaptiveMaxPool2DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool2DGrad, BaseOperator);

// AG means auto generated
class MIND_API AGAdaptiveMaxPool2DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveMaxPool2DGrad, prim::kPrimAdaptiveMaxPool2DGrad, AGAdaptiveMaxPool2DGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
