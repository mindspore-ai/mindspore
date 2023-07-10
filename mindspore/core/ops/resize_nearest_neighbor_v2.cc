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

#include "ops/resize_nearest_neighbor_v2.h"

#include <map>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/format.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/image_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ResizeNearestNeighborV2InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  int64_t long_kdim1 = static_cast<int64_t>(kDim1);
  int64_t long_kdim2 = static_cast<int64_t>(kDim2);
  int64_t long_kdim4 = static_cast<int64_t>(kDim4);
  (void)CheckAndConvertUtils::CheckInteger("dimension of size", SizeToLong(size_shape.size()), kEqual, long_kdim1,
                                           prim_name);

  auto align_corners_ptr = primitive->GetAttr(kAlignCorners);
  MS_EXCEPTION_IF_NULL(align_corners_ptr);
  auto align_corners = GetValue<bool>(align_corners_ptr);
  auto half_pixel_centers_ptr = primitive->GetAttr(kHalfPixelCenters);
  MS_EXCEPTION_IF_NULL(half_pixel_centers_ptr);
  auto half_pixel_centers = GetValue<bool>(half_pixel_centers_ptr);
  if (align_corners && half_pixel_centers) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << ". If half_pixel_centers is True, align_corners must be False.";
  }

  ShapeVector y_shape(long_kdim4, abstract::Shape::kShapeDimAny);
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of x", SizeToLong(x_shape.size()), kEqual, long_kdim4,
                                             prim_name);
    y_shape[kInputIndex0] = x_shape[kInputIndex0];
    y_shape[kInputIndex1] = x_shape[kInputIndex1];
  }

  auto size_value = GetShapeValue(primitive, input_args[kInputIndex1]);
  if (!IsDynamic(size_value)) {
    if (size_value.size() != static_cast<size_t>(long_kdim2)) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the elements number of 'size' should be 2, but get "
                               << size_value.size() << " number.";
    }
    (void)CheckAndConvertUtils::CheckPositiveVector("size", size_value, prim_name);
    y_shape[kInputIndex2] = size_value.front();
    y_shape[kInputIndex3] = size_value.back();
  }

  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr ResizeNearestNeighborV2InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> support_types = {kUInt8, kFloat16, kFloat32, kFloat64};
  auto start_type = CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(),
                                                               support_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                                   primitive->name());
  return start_type;
}
}  // namespace

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

AbstractBasePtr ResizeNearestNeighborV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_type = ResizeNearestNeighborV2InferType(primitive, input_args);
  auto infer_shape = ResizeNearestNeighborV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ResizeNearestNeighborV2, BaseOperator);

// AG means auto generated
class MIND_API AGResizeNearestNeighborV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborV2, prim::kPrimResizeNearestNeighborV2,
                                 AGResizeNearestNeighborV2Infer, false);
}  // namespace ops
}  // namespace mindspore
