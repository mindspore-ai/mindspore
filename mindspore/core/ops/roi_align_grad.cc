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

#include <string>
#include <memory>
#include "ops/roi_align_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr auto kInputNumNoShape = 2;
constexpr auto kInputNumWithShape = 3;
MIND_API_OPERATOR_IMPL(ROIAlignGrad, BaseOperator);
void ROIAlignGrad::set_pooled_height(const int64_t pooled_height) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_height));
}

int64_t ROIAlignGrad::get_pooled_height() const { return GetValue<int64_t>(GetAttr(kPooledHeight)); }

void ROIAlignGrad::set_pooled_width(const int64_t pooled_width) {
  (void)this->AddAttr(kPooledWidth, api::MakeValue(pooled_width));
}

int64_t ROIAlignGrad::get_pooled_width() const { return GetValue<int64_t>(GetAttr(kPooledWidth)); }

void ROIAlignGrad::set_spatial_scale(const float spatial_scale) {
  (void)this->AddAttr(kSpatialScale, api::MakeValue(spatial_scale));
}

float ROIAlignGrad::get_spatial_scale() const { return GetValue<float>(GetAttr(kSpatialScale)); }

void ROIAlignGrad::set_sample_num(const int64_t sample_num) {
  (void)this->AddAttr(kSampleNum, api::MakeValue(sample_num));
}

int64_t ROIAlignGrad::get_sample_num() const { return GetValue<int64_t>(GetAttr(kSampleNum)); }

void ROIAlignGrad::Init(const int64_t pooled_height, const int64_t pooled_width, const float spatial_scale,
                        const int64_t sample_num) {
  this->set_pooled_height(pooled_height);
  this->set_pooled_width(pooled_width);
  this->set_spatial_scale(spatial_scale);
  this->set_sample_num(sample_num);
}

TypePtr ROIAlignGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return input_args[kInputIndex0]->BuildType();
}

abstract::ShapePtr ROIAlignGradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  ShapeVector out_shape;
  if (input_args.size() == static_cast<size_t>(kInputNumWithShape)) {
    auto input_shape = input_args[kInputIndex2];
    out_shape = GetShapeValue(primitive, input_shape);
  } else {
    auto input_shape_attr = primitive->GetAttr(kInputShape);
    MS_EXCEPTION_IF_NULL(input_shape_attr);
    out_shape = GetValue<ShapeVector>(input_shape_attr);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

abstract::AbstractBasePtr ROIAlignGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInRange("the number of inputs", input_args.size(), kIncludeBoth,
                                     {kInputNumNoShape, kInputNumWithShape}, primitive->name());
  auto types = ROIAlignGradInferType(primitive, input_args);
  auto shapes = ROIAlignGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameROIAlignGrad, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(ROIAlignGrad, prim::kPrimROIAlignGrad, ROIAlignGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
