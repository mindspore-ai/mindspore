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
#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ROIAlign, BaseOperator);
class ROIAlignInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t kInputNum = 2;
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, op_name);

    auto feature_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    int64_t out_c, out_n;
    if (IsDynamicRank(feature_shape)) {
      out_c = abstract::Shape::kShapeDimAny;
    } else {
      constexpr size_t kFeatureShapeSize = 4;
      (void)CheckAndConvertUtils::CheckInteger("rank of feature shape", SizeToLong(feature_shape.size()), kLessEqual,
                                               kFeatureShapeSize, op_name);
      out_c = feature_shape[kInputIndex1];
    }
    if (IsDynamicRank(rois_shape)) {
      out_n = abstract::Shape::kShapeDimAny;
    } else {
      constexpr size_t kRoisShapeSize = 2;
      (void)CheckAndConvertUtils::CheckInteger("rank of rois shape", SizeToLong(rois_shape.size()), kEqual,
                                               kRoisShapeSize, op_name);
      auto rois_second_dim = rois_shape[kInputIndex1];
      if (rois_second_dim != abstract::Shape::kShapeDimAny) {
        constexpr int64_t kRoisShapeSecondDim = 5;
        (void)CheckAndConvertUtils::CheckInteger("second dim of rois shape", rois_second_dim, kEqual,
                                                 kRoisShapeSecondDim);
      }
      out_n = rois_shape[kInputIndex0];
    }
    ShapeVector output_shape;
    auto pooled_height_ptr = primitive->GetAttr(kPooledHeight);
    MS_EXCEPTION_IF_NULL(pooled_height_ptr);
    auto pooled_height = GetValue<int64_t>(pooled_height_ptr);

    auto pooled_width_ptr = primitive->GetAttr(kPooledWidth);
    MS_EXCEPTION_IF_NULL(pooled_width_ptr);
    auto pooled_width = GetValue<int64_t>(pooled_width_ptr);

    output_shape.emplace_back(out_n);
    output_shape.emplace_back(out_c);
    output_shape.emplace_back(pooled_height);
    output_shape.emplace_back(pooled_width);

    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto op_name = prim->name();
    auto feature_type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(feature_type);
    auto rois_type = input_args[kInputIndex1]->BuildType();
    MS_EXCEPTION_IF_NULL(rois_type);
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("feature", feature_type, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", rois_type, valid_types, op_name);
    return feature_type;
  }
};

void ROIAlign::set_pooled_height(const int64_t pooled_height) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_height));
}

int64_t ROIAlign::get_pooled_height() const {
  auto value_ptr = GetAttr(kPooledHeight);
  return static_cast<int64_t>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_pooled_width(const int64_t pooled_width) {
  (void)this->AddAttr(kPooledHeight, api::MakeValue(pooled_width));
}

int64_t ROIAlign::get_pooled_width() const {
  auto value_ptr = GetAttr(kPooledWidth);
  return static_cast<int64_t>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_spatial_scale(const float spatial_scale) {
  (void)this->AddAttr(kSpatialScale, api::MakeValue(spatial_scale));
}

float ROIAlign::get_spatial_scale() const {
  auto value_ptr = GetAttr(kSpatialScale);
  return GetValue<float>(value_ptr);
}

void ROIAlign::set_sample_num(const int64_t sample_num) { (void)this->AddAttr(kSampleNum, api::MakeValue(sample_num)); }

int64_t ROIAlign::get_sample_num() const {
  auto value_ptr = GetAttr(kSampleNum);
  return static_cast<int64_t>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::set_roi_end_mode(const int64_t roi_end_mode) {
  (void)this->AddAttr(kRoiEndMode, api::MakeValue(roi_end_mode));
}

int64_t ROIAlign::get_roi_end_mode() const {
  auto value_ptr = GetAttr(kRoiEndMode);
  return static_cast<int64_t>(GetValue<int64_t>(value_ptr));
}

void ROIAlign::Init(const int64_t pooled_height, const int64_t pooled_weight, const float spatial_scale,
                    const int64_t sample_num, const int64_t roi_end_mode) {
  this->set_pooled_height(pooled_height);
  this->set_pooled_width(pooled_weight);
  this->set_spatial_scale(spatial_scale);
  this->set_sample_num(sample_num);
  this->set_roi_end_mode(roi_end_mode);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(ROIAlign, prim::kPrimROIAlign, ROIAlignInfer, false);
}  // namespace ops
}  // namespace mindspore
