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

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kFeatureShapeSize = 4;
constexpr size_t kRoisShapeSize = 2;
constexpr int64_t kRoisShapeSecondDim = 5;
constexpr size_t kInputSizeNum = 2;
abstract::ShapePtr ROIAlignInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto feature_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("rank of feature shape", SizeToLong(feature_shape.size()), kEqual,
                                           kFeatureShapeSize, op_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of rois shape", SizeToLong(rois_shape.size()), kEqual, kRoisShapeSize,
                                           op_name);

  auto rois_second_dim = rois_shape[kInputIndex1];
  (void)CheckAndConvertUtils::CheckInteger("second dim of rois shape", rois_second_dim, kEqual, kRoisShapeSecondDim);

  ShapeVector output_shape;
  auto pooled_height_ptr = primitive->GetAttr(kPooledHeight);
  MS_EXCEPTION_IF_NULL(pooled_height_ptr);
  auto pooled_height = GetValue<int64_t>(pooled_height_ptr);

  auto pooled_width_ptr = primitive->GetAttr(kPooledWidth);
  MS_EXCEPTION_IF_NULL(pooled_width_ptr);
  auto pooled_width = GetValue<int64_t>(pooled_width_ptr);

  output_shape.emplace_back(rois_shape[0]);
  output_shape.emplace_back(feature_shape[1]);
  output_shape.emplace_back(pooled_height);
  output_shape.emplace_back(pooled_width);

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ROIAlignInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto feature_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(feature_type);
  auto rois_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(rois_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rois type", rois_type, valid_types, prim->name());
  return feature_type;
}
}  // namespace

void ROIAlign::set_pooled_height(const int pooled_height) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_height));
}

int ROIAlign::get_pooled_height() const {
  auto value_ptr = GetAttr(kPooledHeight);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_pooled_width(const int pooled_width) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_width));
}

int ROIAlign::get_pooled_width() const {
  auto value_ptr = GetAttr(kPooledWidth);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_spatial_scale(const float spatial_scale) {
  (void)this->AddAttr(kSpatialScale, api::MakeValue(spatial_scale));
}

float ROIAlign::get_spatial_scale() const {
  auto value_ptr = GetAttr(kSpatialScale);
  return GetValue<float>(value_ptr);
}
void ROIAlign::set_sample_num(const int sample_num) { (void)this->AddAttr(kSampleNum, api::MakeValue(sample_num)); }

int ROIAlign::get_sample_num() const {
  auto value_ptr = GetAttr(kSampleNum);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_roi_end_mode(const int roi_end_mode) {
  (void)this->AddAttr(kRoiEndMode, api::MakeValue(roi_end_mode));
}

int ROIAlign::get_roi_end_mode() const {
  auto value_ptr = GetAttr(kRoiEndMode);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::Init(const int pooled_height, const int pooled_weight, const float spatial_scale, const int sample_num,
                    const int roi_end_mode) {
  this->set_pooled_height(pooled_height);
  this->set_pooled_width(pooled_weight);
  this->set_spatial_scale(spatial_scale);
  this->set_sample_num(sample_num);
  this->set_roi_end_mode(roi_end_mode);
}
MIND_API_OPERATOR_IMPL(ROIAlign, BaseOperator);
AbstractBasePtr ROIAlignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kInputSizeNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto infer_type = ROIAlignInferType(primitive, input_args);
  auto infer_shape = ROIAlignInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ROIAlign, prim::kPrimROIAlign, ROIAlignInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
