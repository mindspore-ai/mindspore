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

#include "ops/grad/adaptive_avg_pool_3d_grad.h"

#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
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
abstract::ShapePtr AdaptiveAvgPool3DGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto orig_input_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];

  auto is_dynamic_rank = IsDynamicRank(input_grad_shape) || IsDynamicRank(orig_input_shape_shape);
  if (is_dynamic_rank) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  const int64_t kMinDim = 4;
  const int64_t kMaxDim = 5;
  const size_t input_grad_dims = input_grad_shape.size();
  CheckAndConvertUtils::CheckInRange("rank of input_grad", SizeToLong(input_grad_dims), kIncludeBoth,
                                     {kMinDim, kMaxDim}, prim_name);
  CheckAndConvertUtils::CheckInteger("rank of orig_input_shape", SizeToLong(orig_input_shape_shape.size()), kEqual, 1,
                                     prim_name);
  if (!IsDynamic(orig_input_shape_shape)) {
    CheckAndConvertUtils::CheckInteger("length of orig_input_shape", orig_input_shape_shape[0], kEqual, input_grad_dims,
                                       prim_name);
  }

  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex1]->BuildValue()->isa<tensor::Tensor>()) {
    ShapeVector output_shape = input_grad_shape;
    auto value_ptr = input_args[kInputIndex1]->BuildValue();
    MS_EXCEPTION_IF_NULL(value_ptr);
    auto value = CheckAndConvertUtils::CheckTensorIntValue("origin input shape", value_ptr, prim_name);
    for (int64_t i = 0; i < orig_input_shape_shape[0]; ++i) {
      output_shape[i] = value[i] > 0 ? value[i] : static_cast<int64_t>(1);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    ShapeVector dx_shape(input_grad_dims, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(dx_shape);
  }
}

TypePtr AdaptiveAvgPool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_grad_dtype = input_args[0]->BuildType();
  auto orig_input_shape_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> input_grad_valid = {kInt8, kInt16, kInt32, kInt64, kUInt8, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> orig_inputs_valid = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad", input_grad_dtype, input_grad_valid, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("orig_input_shape", orig_input_shape_dtype, orig_inputs_valid,
                                                   prim_name);
  return input_grad_dtype;
}
}  // namespace

AbstractBasePtr AdaptiveAvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveAvgPool3DGradInferType(primitive, input_args);
  auto shapes = AdaptiveAvgPool3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(AdaptiveAvgPool3DGrad, BaseOperator);
// AG means auto generated
class MIND_API AGAdaptiveAvgPool3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool3DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool3DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveAvgPool3DGrad, prim::kPrimAdaptiveAvgPool3DGrad, AGAdaptiveAvgPool3DGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
