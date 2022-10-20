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

#include "ops/adjust_contrastv2.h"
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AdjustContrastv2InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_images_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  // support dynamic rank and dynamic shape.
  if (IsDynamic(input_images_shape)) {
    return std::make_shared<abstract::Shape>(input_images_shape);
  }
  auto input_contrast_factor_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  const int64_t min_images_dim = 3;
  const int64_t contrast_factor_dim = 0;
  (void)CheckAndConvertUtils::CheckInteger("dimension of AdjustContrastv2 input images",
                                           SizeToLong(input_images_shape.size()), kGreaterEqual, min_images_dim,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("last dimension of AdjustContrastv2 input images",
                                           input_images_shape[input_images_shape.size() - 1], kEqual, min_images_dim,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of AdjustContrastv2 input contrast_factor",
                                           SizeToLong(input_contrast_factor_shape.size()), kEqual, contrast_factor_dim,
                                           prim_name);
  return std::make_shared<abstract::Shape>(input_images_shape);
}

TypePtr AdjustContrastv2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto input_images_type = input_args[0]->BuildType();
  auto input_contrast_factor_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(input_images_type);
  MS_EXCEPTION_IF_NULL(input_contrast_factor_type);
  const std::set<TypePtr> valid_images_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_images_type, valid_images_types, prim_name);
  const std::set<TypePtr> valid_contrast_factor_types = {kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("contrast_factor", input_contrast_factor_type,
                                                   valid_contrast_factor_types, prim_name);
  return input_images_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdjustContrastv2, BaseOperator);
AbstractBasePtr AdjustContrastv2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = AdjustContrastv2InferType(primitive, input_args);
  auto infer_shape = AdjustContrastv2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AdjustContrastv2, prim::kPrimAdjustContrastv2, AdjustContrastv2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
