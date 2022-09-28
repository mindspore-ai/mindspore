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

#include "ops/roi_align.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ROIAlign, BaseOperator);
void ROIAlign::set_pooled_height(const int64_t pooled_height) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_height));
}

int64_t ROIAlign::get_pooled_height() const { return GetValue<int64_t>(GetAttr(kPooledHeight)); }

void ROIAlign::set_pooled_width(const int64_t pooled_width) {
  (void)this->AddAttr(kPooledWidth, api::MakeValue(pooled_width));
}

int64_t ROIAlign::get_pooled_width() const { return GetValue<int64_t>(GetAttr(kPooledWidth)); }

void ROIAlign::set_spatial_scale(const float spatial_scale) {
  (void)this->AddAttr(kSpatialScale, api::MakeValue(spatial_scale));
}

float ROIAlign::get_spatial_scale() const { return GetValue<float>(GetAttr(kSpatialScale)); }

void ROIAlign::set_sample_num(const int64_t sample_num) { (void)this->AddAttr(kSampleNum, api::MakeValue(sample_num)); }

int64_t ROIAlign::get_sample_num() const { return GetValue<int64_t>(GetAttr(kSampleNum)); }

void ROIAlign::set_roi_end_mode(const int64_t roi_end_mode) {
  (void)this->AddAttr(kRoiEndMode, api::MakeValue(roi_end_mode));
}

int64_t ROIAlign::get_roi_end_mode() const { return GetValue<int64_t>(GetAttr(kRoiEndMode)); }

void ROIAlign::Init(const int64_t pooled_height, const int64_t pooled_width, const float spatial_scale,
                    const int64_t sample_num, const int64_t roi_end_mode) {
  this->set_pooled_height(pooled_height);
  this->set_pooled_width(pooled_width);
  this->set_spatial_scale(spatial_scale);
  this->set_spatial_scale(sample_num);
  this->set_spatial_scale(roi_end_mode);
}

TypePtr ROIAlignInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  std::set<TypePtr> valid_dtypes = {kFloat16, kFloat32};
  // the first input, features
  auto features_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("features", features_type, valid_dtypes, prim_name);
  // the second input, rois
  auto rois_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", rois_type, valid_dtypes, prim_name);
  return features_type;
}

abstract::ShapePtr ROIAlignInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // compute the first dim of output shape
  auto features_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  int64_t out_c;
  if (IsDynamicRank(features_shape)) {
    out_c = UNKNOWN_DIM;
  } else {
    auto features_rank = SizeToLong(features_shape.size());
    const int64_t features_rank_required = 4;
    (void)CheckAndConvertUtils::CheckInteger("rank of 'features'", features_rank, kLessEqual, features_rank_required,
                                             prim_name);
    const int64_t channel_index = 1;
    out_c = features_shape[channel_index];
  }
  // compute the second dim of output shape
  auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  int64_t out_n;
  if (IsDynamicRank(rois_shape)) {
    out_n = UNKNOWN_DIM;
  } else {
    auto rois_rank = SizeToLong(rois_shape.size());
    const int64_t rois_rank_required = 2;
    (void)CheckAndConvertUtils::CheckInteger("rank of 'rois'", rois_rank, kEqual, rois_rank_required, prim_name);
    const int64_t roi_num_index = 0;
    out_n = rois_shape[roi_num_index];
  }
  // compute the last two dims of output shape
  auto out_h_ptr = primitive->GetAttr(kPooledHeight);
  MS_EXCEPTION_IF_NULL(out_h_ptr);
  int64_t out_h = GetValue<int64_t>(out_h_ptr);
  auto out_w_ptr = primitive->GetAttr(kPooledWidth);
  MS_EXCEPTION_IF_NULL(out_w_ptr);
  int64_t out_w = GetValue<int64_t>(out_w_ptr);
  std::vector<int64_t> output_shape = {out_n, out_c, out_h, out_w};
  return std::make_shared<abstract::Shape>(output_shape);
}

abstract::AbstractBasePtr ROIAlignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = ROIAlignInferType(primitive, input_args);
  auto shapes = ROIAlignInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ROIAlign, prim::kPrimROIAlign, ROIAlignInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
