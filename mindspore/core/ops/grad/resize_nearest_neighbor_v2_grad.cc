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

#include "ops/grad/resize_nearest_neighbor_v2_grad.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

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
#include "mindspore/core/ops/image_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ResizeNearestNeighborV2GradInferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
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
  if (!IsDynamicRank(grads_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of grads", SizeToLong(grads_shape.size()), kEqual, long_kdim4,
                                             prim_name);
    y_shape[kInputIndex0] = grads_shape[kInputIndex0];
    y_shape[kInputIndex1] = grads_shape[kInputIndex1];
  }

  auto size_ptr = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_ptr);
  if (IsValueKnown(size_ptr)) {
    auto size_value = CheckAndConvertUtils::CheckTensorIntValue("input size", size_ptr, prim_name);
    if (size_value.size() != static_cast<size_t>(long_kdim2)) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the elements number of 'size' should be 2, but get "
                               << size_value.size() << " number.";
    }
    y_shape[kInputIndex2] = size_value.front();
    y_shape[kInputIndex3] = size_value.back();
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr ResizeNearestNeighborV2GradInferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  std::set<TypePtr> support_types = {kUInt8, kFloat16, kFloat32, kFloat64};
  auto grads_type = CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[kInputIndex0]->BuildType(),
                                                               support_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                                   prim_name);
  return grads_type;
}
}  // namespace

bool ResizeNearestNeighborV2Grad::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

bool ResizeNearestNeighborV2Grad::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

std::string ResizeNearestNeighborV2Grad::get_data_format() const {
  auto value_ptr = GetAttr(kFormat);
  return GetValue<std::string>(value_ptr);
}

AbstractBasePtr ResizeNearestNeighborV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto infer_shape = ResizeNearestNeighborV2GradInferShape(primitive, input_args);
  auto infer_type = ResizeNearestNeighborV2GradInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ResizeNearestNeighborV2Grad, BaseOperator);

// AG means auto generated
class MIND_API AGResizeNearestNeighborV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborV2GradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborV2Grad, prim::kPrimResizeNearestNeighborV2Grad,
                                 AGResizeNearestNeighborV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
