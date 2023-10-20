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
#include "ops/resize_bicubic.h"
#include "ops/resize_bilinear_v2.h"
#include "ops/resize_nearest_neighbor_v2.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/image_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ResizeBicubic, BaseOperator);
void ResizeBicubic::set_align_corners(const bool align_corners) {
  (void)this->AddAttr("align_corners", api::MakeValue(align_corners));
}
void ResizeBicubic::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers));
}

bool ResizeBicubic::get_align_corners() const {
  auto value_ptr = GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}
bool ResizeBicubic::get_half_pixel_centers() const {
  auto value_ptr = GetAttr("half_pixel_centers");
  return GetValue<bool>(value_ptr);
}

void ResizeBicubic::Init(const bool align_corners, const bool half_pixel_centers) {
  this->set_align_corners(align_corners);
  this->set_half_pixel_centers(half_pixel_centers);
}

MIND_API_OPERATOR_IMPL(ResizeNearestNeighborV2, BaseOperator);
bool ResizeNearestNeighborV2::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

bool ResizeNearestNeighborV2::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

std::string ResizeNearestNeighborV2::get_data_format() const {
  auto value_ptr = GetAttr(kFormat);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ResizeBilinearV2, BaseOperator);
void ResizeBilinearV2::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

bool ResizeBilinearV2::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

void ResizeBilinearV2::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr(kHalfPixelCenters, api::MakeValue(half_pixel_centers));
}

bool ResizeBilinearV2::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

namespace {
constexpr int64_t resize_2d_input_num = 2;
void AttrTest(bool a, bool b) {
  if (a && b) {
    MS_EXCEPTION(ValueError) << "The half_pixel_centers must be false when align_corners is true "
                             << ", but half_pixel_centers got True";
  }
}

abstract::ShapePtr Resize2DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, resize_2d_input_num, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto align_corners_ptr = primitive->GetAttr("align_corners");
  bool align_corners = GetValue<bool>(align_corners_ptr);
  auto half_pixel_centers_ptr = primitive->GetAttr("half_pixel_centers");
  bool half_pixel_centers = GetValue<bool>(half_pixel_centers_ptr);
  AttrTest(align_corners, half_pixel_centers);

  const int64_t x_rank = 4;
  std::vector<int64_t> output_shape(x_rank, abstract::Shape::kShapeDimAny);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("images' rank", SizeToLong(x_shape.size()), kEqual, x_rank, prim_name);
    output_shape[kInputIndex0] = x_shape[kInputIndex0];
    output_shape[kInputIndex1] = x_shape[kInputIndex1];
  }

  auto size_value = GetShapeValue(primitive, input_args[kInputIndex1]);
  if (IsValueKnown(input_args[kInputIndex1]->BuildValue()) &&
      std::any_of(size_value.begin(), size_value.end(), [](auto x) { return x < 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the value of 'size' must not be negative, but got ["
                             << size_value[kIndex0] << ", " << size_value[kIndex1] << "].";
  }
  if (IsDynamicRank(size_value)) {
    size_value = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
  }
  const int64_t size_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("the dimension of size", SizeToLong(size_value.size()), kEqual, size_size,
                                           prim_name);
  if (!IsDynamic(size_value)) {
    const int64_t kNumZero = 0;
    for (size_t i = 0; i < size_value.size(); ++i) {
      (void)CheckAndConvertUtils::CheckInteger("size", size_value[i], kGreaterThan, kNumZero, prim_name);
    }
  }
  output_shape[kInputIndex2] = size_value[kInputIndex0];
  output_shape[kInputIndex3] = size_value[kInputIndex1];

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr Resize2DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, resize_2d_input_num, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_type = input_args[kInputIndex0]->BuildType();
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto size_type = input_args[kInputIndex1]->BuildType();
  if (prim_name == kNameResizeNearestNeighborV2) {
    (void)valid_types.insert(kUInt8);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, {kInt32, kInt64}, prim_name);
  } else if (prim_name == kNameResizeBicubic) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, {kInt32}, prim_name);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

// AG means auto generated
class MIND_API AGResize2DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Resize2DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Resize2DInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto infer_type = Resize2DInferType(primitive, input_args);
    auto infer_shape = Resize2DInferShape(primitive, input_args);
    return abstract::MakeAbstractTensor(infer_shape, infer_type);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeBicubic, prim::kPrimResizeBicubic, AGResize2DInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborV2, prim::kPrimResizeNearestNeighborV2, AGResize2DInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeBilinearV2, prim::kPrimResizeBilinearV2, AGResize2DInfer, false);
}  // namespace ops
}  // namespace mindspore
