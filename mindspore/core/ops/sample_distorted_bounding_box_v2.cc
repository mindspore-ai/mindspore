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

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
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

  auto aspect_ratio_range = GetValue<std::vector<float>>(primitive->GetAttr("aspect_ratio_range"));
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

  auto area_range = GetValue<std::vector<float>>(primitive->GetAttr("area_range"));
  auto area_range_dim = area_range.size();
  (void)CheckAndConvertUtils::CheckInteger("area_range elements", SizeToLong(area_range_dim), kEqual, kSize2,
                                           prim_name);
  for (size_t i = 0; i < area_range_dim; ++i) {
    (void)CheckAndConvertUtils::CheckInRange<float>("area_range value", area_range[i], kIncludeRight, {0.0, 1.0},
                                                    prim_name);
  }
  if (area_range[kIndex0] >= area_range[kIndex1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', area_range[0] must less than area_range[1].";
  }

  auto use_image_if_no_bounding_boxes = GetValue<bool>(primitive->GetAttr("use_image_if_no_bounding_boxes"));
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

MIND_API_OPERATOR_IMPL(SampleDistortedBoundingBoxV2, BaseOperator);
AbstractBasePtr SampleDistortedBoundingBoxV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = SampleDistortedBoundingBoxV2InferType(primitive, input_args);
  auto infer_shape = SampleDistortedBoundingBoxV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(SampleDistortedBoundingBoxV2, prim::kPrimSampleDistortedBoundingBoxV2,
                             SampleDistortedBoundingBoxV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
