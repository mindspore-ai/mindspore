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

#include "ops/grad/adaptive_avg_pool_2d_grad.h"

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
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AdaptiveAvgPool2DGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto orig_input_shape = GetShapeValue(primitive, input_args[1]);
  if (!IsDynamicRank(orig_input_shape)) {
    const int64_t orig_input_shape_shape = SizeToLong(orig_input_shape.size());
    (void)CheckAndConvertUtils::CheckInRange("length of orig_input_shape", orig_input_shape_shape, kIncludeBoth, {3, 4},
                                             kNameAdaptiveAvgPool2DGrad);
  }
  return std::make_shared<abstract::Shape>(orig_input_shape);
}

TypePtr AdaptiveAvgPool2DGradInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> input_grad_valid = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad", input_grad_dtype, input_grad_valid,
                                                   kNameAdaptiveAvgPool2DGrad);
  return input_grad_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveAvgPool2DGrad, BaseOperator);
AbstractBasePtr AdaptiveAvgPool2DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveAvgPool2DGradInferType(primitive, input_args);
  auto shapes = AdaptiveAvgPool2DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGAdaptiveAvgPool2DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool2DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool2DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveAvgPool2DGradInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveAvgPool2DGrad, prim::kPrimAdaptiveAvgPool2DGrad, AGAdaptiveAvgPool2DGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
