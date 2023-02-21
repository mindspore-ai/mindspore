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
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ROIAlignGrad, BaseOperator);
class ROIAlignGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    std::vector<int64_t> output_shape;
    auto feature_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    if (!IsDynamicRank(feature_shape)) {
      constexpr size_t kROIGradFeatureShapeSize = 4;
      (void)CheckAndConvertUtils::CheckInteger("rank of feature shape", SizeToLong(feature_shape.size()), kLessEqual,
                                               kROIGradFeatureShapeSize, op_name);
    }
    if (!IsDynamicRank(rois_shape)) {
      constexpr size_t kROIGradRoisShapeSize = 2;
      (void)CheckAndConvertUtils::CheckInteger("rank of rois shape", SizeToLong(rois_shape.size()), kEqual,
                                               kROIGradRoisShapeSize, op_name);
    }
    constexpr size_t kInputNum = 3;
    if (input_args.size() == kInputNum) {
      auto input_shape = input_args[kInputIndex2];
      output_shape = GetShapeValue(primitive, input_shape);
      return std::make_shared<abstract::Shape>(output_shape);
    } else if (input_args.size() == kInputNum - 1) {
      auto input_shape = primitive->GetAttr("xdiff_shape");
      MS_EXCEPTION_IF_NULL(input_shape);
      auto input_shape_tuple = input_shape->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(input_shape_tuple);
      auto input_tuple = input_shape_tuple->value();
      (void)std::transform(input_tuple.begin(), input_tuple.end(), std::back_inserter(output_shape),
                           [&op_name, &input_shape](const ValuePtr &size_value) -> int64_t {
                             if (!size_value->isa<Int64Imm>()) {
                               MS_EXCEPTION(TypeError)
                                 << "For primitive[" << op_name << "], the 'shape'"
                                 << " must be a tuple with all Int elements, but got " << input_shape->ToString();
                             }
                             return GetValue<int64_t>(size_value);
                           });
    } else {
      MS_EXCEPTION(TypeError) << "For primitive[" << op_name << "], the 'input num'"
                              << " must be 2 or 3, but got " << input_args.size();
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("ydiff", input_args[kInputIndex0]->BuildType(), valid_types,
                                                     prim->name());
    (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", input_args[kInputIndex1]->BuildType(), valid_types,
                                                     prim->name());
    return input_args[kInputIndex0]->BuildType();
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

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
REGISTER_PRIMITIVE_OP_INFER_IMPL(ROIAlignGrad, prim::kPrimROIAlignGrad, ROIAlignGradInfer, false);
}  // namespace ops
}  // namespace mindspore
