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

#include "ops/adjust_saturation.h"

#include <memory>
#include <vector>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
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
abstract::ShapePtr AdjustSaturationInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_image_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  // support dynamic rank and dynamic shape.
  if (IsDynamic(input_image_shape)) {
    return std::make_shared<abstract::Shape>(input_image_shape);
  }
  auto input_scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  const int64_t min_image_dim = 3;
  const int64_t scale_dim = 0;
  (void)CheckAndConvertUtils::CheckInteger("dimension of AdjustSaturation input image",
                                           SizeToLong(input_image_shape.size()), kGreaterEqual, min_image_dim,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("last dimension of AdjustSaturation input image",
                                           input_image_shape[input_image_shape.size() - 1], kEqual, min_image_dim,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of AdjustSaturation input scale",
                                           SizeToLong(input_scale_shape.size()), kEqual, scale_dim, prim_name);
  return std::make_shared<abstract::Shape>(input_image_shape);
}

TypePtr AdjustSaturationInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto input_images_type = input_args[0]->BuildType();
  auto input_scale_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(input_images_type);
  MS_EXCEPTION_IF_NULL(input_scale_type);
  const std::set<TypePtr> valid_images_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("image", input_images_type, valid_images_types, prim_name);
  const std::set<TypePtr> valid_scale_types = {kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scale", input_scale_type, valid_scale_types, prim_name);
  return input_images_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdjustSaturation, BaseOperator);
AbstractBasePtr AdjustSaturationInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = AdjustSaturationInferType(primitive, input_args);
  auto infer_shape = AdjustSaturationInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGAdjustSaturationInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdjustSaturationInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdjustSaturationInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdjustSaturationInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdjustSaturation, prim::kPrimAdjustSaturation, AGAdjustSaturationInfer, false);
}  // namespace ops
}  // namespace mindspore
