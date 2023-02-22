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

#include <set>

#include "ops/sample_distorted_bounding_box_v2.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kAspectRatioRange = "aspect_ratio_range";
constexpr auto kAreaRange = "area_range";
constexpr auto kMaxAttempts = "max_attempts";
constexpr auto kUseImage = "use_image_if_no_bounding_boxes";

abstract::TupleShapePtr SampleDistortedBoundingBoxV2InferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const constexpr int64_t kIndex0 = 0;
  const constexpr int64_t kIndex1 = 1;
  const constexpr int64_t kIndex2 = 2;
  const constexpr int64_t kSize1 = 1;
  const constexpr int64_t kSize2 = 2;
  const constexpr int64_t kSize3 = 3;
  const constexpr int64_t kSize4 = 4;

  auto image_size_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto image_size_dim = SizeToLong(image_size_shape.size());
  auto bboxes_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto bboxes_dim = SizeToLong(bboxes_shape.size());
  auto min_object_covered_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto min_object_covered_dim = SizeToLong(min_object_covered_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("image_size dimension", image_size_dim, kEqual, kSize1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("image_size elements", image_size_shape[kIndex0], kEqual, kSize3, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("bounding_boxes dimension", bboxes_dim, kEqual, kSize3, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("elements of each bounding box in bounding_boxes", bboxes_shape[kIndex2],
                                           kEqual, kSize4, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("min_object_covered dimension", min_object_covered_dim, kEqual, kSize1,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("min_object_covered elements", min_object_covered_shape[kIndex0], kEqual,
                                           kSize1, prim_name);

  auto aspect_ratio_range = GetValue<std::vector<float>>(primitive->GetAttr(kAspectRatioRange));
  auto aspect_ratio_range_dim = aspect_ratio_range.size();
  (void)CheckAndConvertUtils::CheckInteger("aspect_ratio_range elements", SizeToLong(aspect_ratio_range_dim), kEqual,
                                           kSize2, prim_name);
  for (size_t i = 0; i < aspect_ratio_range_dim; ++i) {
    if (aspect_ratio_range[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', aspect_ratio_range must be positive.";
    }
  }
  if (aspect_ratio_range[kIndex0] >= aspect_ratio_range[kIndex1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', aspect_ratio_range[0] must less than aspect_ratio_range[1].";
  }

  auto area_range = GetValue<std::vector<float>>(primitive->GetAttr(kAreaRange));
  auto area_range_dim = area_range.size();
  (void)CheckAndConvertUtils::CheckInteger("area_range elements", SizeToLong(area_range_dim), kEqual, kSize2,
                                           prim_name);
  for (size_t i = 0; i < area_range_dim; ++i) {
    CheckAndConvertUtils::CheckInRange<float>("area_range value", area_range[i], kIncludeRight, {0.0, 1.0}, prim_name);
  }
  if (area_range[kIndex0] >= area_range[kIndex1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', area_range[0] must less than area_range[1].";
  }

  auto use_image_if_no_bounding_boxes = GetValue<bool>(primitive->GetAttr(kUseImage));
  if (!use_image_if_no_bounding_boxes) {
    if (bboxes_shape[kIndex0] == 0 || bboxes_shape[kIndex1] == 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', batch or N in bounding_boxes whose shape is [batch, N, 4] equals 0, which means "
                               << "no bounding boxes provided as input. Set use_image_if_no_bounding_boxes=True if you"
                               << " wish to not provide any bounding boxes.";
    }
  }

  std::vector<int64_t> shape, shape_box;
  shape.push_back(kSize3);
  shape_box.push_back(kSize1);
  shape_box.push_back(kSize1);
  shape_box.push_back(kSize4);
  abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(shape);
  abstract::ShapePtr out_shape_box = std::make_shared<abstract::Shape>(shape_box);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{out_shape, out_shape, out_shape_box});
}

TuplePtr SampleDistortedBoundingBoxV2InferType(const PrimitivePtr &prim,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> valid_types1 = {kUInt8, kInt8, kInt16, kInt32, kInt64};
  const std::set<TypePtr> valid_types2 = {kFloat32};
  auto image_size_type = input_args[kInputIndex0]->BuildType();
  auto bboxes_type = input_args[kInputIndex1]->BuildType();
  auto min_object_type = input_args[kInputIndex2]->BuildType();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("image_size", image_size_type, valid_types1, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bounding_boxes", bboxes_type, valid_types2, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("min_object_covered", min_object_type, valid_types2, name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{image_size_type, image_size_type, min_object_type});
}
}  // namespace

void SampleDistortedBoundingBoxV2::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }
int64_t SampleDistortedBoundingBoxV2::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void SampleDistortedBoundingBoxV2::set_seed2(const int64_t seed2) {
  (void)this->AddAttr(kSeed2, api::MakeValue(seed2));
}
int64_t SampleDistortedBoundingBoxV2::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void SampleDistortedBoundingBoxV2::set_aspect_ratio_range(const std::vector<float> aspect_ratio_range) {
  (void)this->AddAttr(kAspectRatioRange, api::MakeValue(aspect_ratio_range));
}
std::vector<float> SampleDistortedBoundingBoxV2::get_aspect_ratio_range() const {
  auto value_ptr = GetAttr(kAspectRatioRange);
  return GetValue<std::vector<float>>(value_ptr);
}
void SampleDistortedBoundingBoxV2::set_area_range(const std::vector<float> area_range) {
  (void)this->AddAttr(kAreaRange, api::MakeValue(area_range));
}
std::vector<float> SampleDistortedBoundingBoxV2::get_area_range() const {
  auto value_ptr = GetAttr(kAreaRange);
  return GetValue<std::vector<float>>(value_ptr);
}
void SampleDistortedBoundingBoxV2::set_max_attempts(const int64_t max_attempts) {
  (void)this->AddAttr(kMaxAttempts, api::MakeValue(max_attempts));
}
int64_t SampleDistortedBoundingBoxV2::get_max_attempts() const {
  auto value_ptr = GetAttr(kMaxAttempts);
  return GetValue<int64_t>(value_ptr);
}
void SampleDistortedBoundingBoxV2::set_use_image(const bool use_image) {
  (void)this->AddAttr(kUseImage, api::MakeValue(use_image));
}
bool SampleDistortedBoundingBoxV2::get_use_image() const {
  auto value_ptr = GetAttr(kUseImage);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(SampleDistortedBoundingBoxV2, BaseOperator);
AbstractBasePtr SampleDistortedBoundingBoxV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = SampleDistortedBoundingBoxV2InferType(primitive, input_args);
  auto infer_shape = SampleDistortedBoundingBoxV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSampleDistortedBoundingBoxV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SampleDistortedBoundingBoxV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SampleDistortedBoundingBoxV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SampleDistortedBoundingBoxV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SampleDistortedBoundingBoxV2, prim::kPrimSampleDistortedBoundingBoxV2,
                                 AGSampleDistortedBoundingBoxV2Infer, false);
}  // namespace ops
}  // namespace mindspore
