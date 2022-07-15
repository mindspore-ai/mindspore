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

#include "ops/grad/roi_align_grad.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kROIGradFeatureShapeSize = 4;
constexpr size_t kROIGradRoisShapeSize = 2;
abstract::ShapePtr ROIAlignGradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto feature_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("rank of feature shape", SizeToLong(feature_shape.size()), kEqual,
                                           kROIGradFeatureShapeSize, op_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of rois shape", SizeToLong(rois_shape.size()), kEqual,
                                           kROIGradRoisShapeSize, op_name);
  auto xdiff_shape_ptr = primitive->GetAttr("xdiff_shape");
  MS_EXCEPTION_IF_NULL(xdiff_shape_ptr);
  auto xdiff_shape = GetValue<std::vector<int64_t>>(xdiff_shape_ptr);
  return std::make_shared<abstract::Shape>(xdiff_shape);
}

TypePtr ROIAlignGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ydiff", input_args[0]->BuildType(), {kFloat32, kFloat16},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", input_args[1]->BuildType(), {kFloat32, kFloat16},
                                                   prim->name());
  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(ROIAlignGrad, BaseOperator);
AbstractBasePtr ROIAlignGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);

  auto type = ROIAlignGradInferType(primitive, input_args);
  auto shape = ROIAlignGradInferShape(primitive, input_args);

  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ROIAlignGrad, prim::kPrimROIAlignGrad, ROIAlignGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
