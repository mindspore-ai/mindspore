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

#include <string>
#include <memory>
#include <map>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
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
namespace {
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->isa<ValueAny>()))
abstract::ShapePtr ResizeNearestNeighborV2GradInferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto size_ptr = input_args[kInputIndex1]->BuildValue();
  int64_t long_kdim1 = static_cast<int64_t>(kDim1);
  int64_t long_kdim2 = static_cast<int64_t>(kDim2);
  int64_t long_kdim4 = static_cast<int64_t>(kDim4);
  (void)CheckAndConvertUtils::CheckInteger("dimension of size", SizeToLong(size_shape.size()), kEqual, long_kdim1,
                                           prim_name);

  auto data_format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr(kFormat));
  std::map<char, size_t> dim_idx_map;
  auto align_corners_ptr = primitive->GetAttr(kAlignCorners);
  MS_EXCEPTION_IF_NULL(align_corners_ptr);
  auto align_corners = GetValue<bool>(align_corners_ptr);
  auto half_pixel_centers_ptr = primitive->GetAttr(kHalfPixelCenters);
  MS_EXCEPTION_IF_NULL(half_pixel_centers_ptr);
  auto half_pixel_centers = GetValue<bool>(half_pixel_centers_ptr);
  if (align_corners && half_pixel_centers) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << ". If half_pixel_centers is True, align_corners must be False.";
  }
  mindspore::Format format_enum = static_cast<mindspore::Format>(data_format);

  if (format_enum == Format::NCHW) {
    dim_idx_map = {{'N', kInputIndex0}, {'C', kInputIndex1}, {'H', kInputIndex2}, {'W', kInputIndex3}};
  } else if (format_enum == Format::NHWC) {
    dim_idx_map = {{'N', kInputIndex0}, {'H', kInputIndex1}, {'W', kInputIndex2}, {'C', kInputIndex3}};
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the attr of 'data_format' only support [" << kFormatNCHW
                             << ", " << kFormatNHWC << "]. But get '" << data_format << "'.";
  }

  ShapeVector y_shape(long_kdim4);
  if (IsDynamicRank(grads_shape)) {
    y_shape[dim_idx_map['N']] = abstract::Shape::kShapeDimAny;
    y_shape[dim_idx_map['C']] = abstract::Shape::kShapeDimAny;
  } else {
    (void)CheckAndConvertUtils::CheckInteger("dimension of grads", SizeToLong(grads_shape.size()), kEqual, long_kdim4,
                                             prim_name);
    y_shape[dim_idx_map['N']] = grads_shape[dim_idx_map['N']];
    y_shape[dim_idx_map['C']] = grads_shape[dim_idx_map['C']];
  }

  bool is_compile = IsNoneOrAnyValue(size_ptr);
  if (is_compile) {
    y_shape[dim_idx_map['H']] = abstract::Shape::kShapeDimAny;
    y_shape[dim_idx_map['W']] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::Shape>(y_shape);
  } else {
    MS_EXCEPTION_IF_NULL(size_ptr);
    auto size_value = CheckAndConvertUtils::CheckTensorIntValue("input size", size_ptr, prim_name);
    if (size_value.size() != static_cast<size_t>(long_kdim2)) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the elements number of 'size' should be 2, but get "
                               << size_value.size() << " number.";
    }
    y_shape[dim_idx_map['H']] = size_value.front();
    y_shape[dim_idx_map['W']] = size_value.back();
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr ResizeNearestNeighborV2GradInferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> support_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64};
  auto grads_type = CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[kInputIndex0]->BuildType(),
                                                               support_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                                   primitive->name());
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
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborV2Grad, prim::kPrimResizeNearestNeighborV2Grad,
                                 AGResizeNearestNeighborV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
