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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "ops/grad/resize_v2_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ResizeV2GradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  int64_t grads_shape_0 = SizeToLong(grad_shape[0]);
  int64_t grads_shape_1 = SizeToLong(grad_shape[1]);
  const int64_t kDimSize = 4;
  (void)CheckAndConvertUtils::CheckInteger("dim of grads", SizeToLong(grad_shape.size()), kEqual, kDimSize, prim_name);

  auto roi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  const int64_t kRoiDimSize = 1;
  (void)CheckAndConvertUtils::CheckInteger("dim of roi", SizeToLong(roi_shape.size()), kEqual, kRoiDimSize, prim_name);

  auto scales_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  const int64_t kScalesDimSize = 1;
  (void)CheckAndConvertUtils::CheckInteger("dim of scales", SizeToLong(scales_shape.size()), kEqual, kScalesDimSize,
                                           prim_name);

  auto sizes_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  const int64_t kSizesDimSize = 1;
  (void)CheckAndConvertUtils::CheckInteger("dim of original_size", SizeToLong(sizes_shape.size()), kEqual,
                                           kSizesDimSize, prim_name);

  auto sizes_input = input_args[kInputIndex3]->BuildValue();
  auto mode_ptr = primitive->GetAttr("mode");
  std::string mode_str = GetValue<std::string>(mode_ptr);

  if (!sizes_input->isa<ValueAny>() && !sizes_input->isa<None>()) {
    MS_EXCEPTION_IF_NULL(sizes_input);
    auto sizes = CheckAndConvertUtils::CheckTensorIntValue("original_size", sizes_input, prim_name);
    const size_t kSizesSize = 4;
    if (sizes.size() != kSizesSize) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the length of 'original_size' must be 4.";
    }
    if (mode_str != "cubic" && sizes[kInputIndex2] != 1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', original_size[2] must be 1, "
                               << "when mode='nearest' or 'linear'.";
    }
    if (sizes[kInputIndex2] <= 0 || sizes[kInputIndex3] <= 0) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of 'original_size' "
                               << "must be greater than 0.";
    }
    if (sizes[kInputIndex0] != grads_shape_0 || sizes[kInputIndex1] != grads_shape_1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', original_size[0] and original_size[1] must be "
                               << "the same as the shape[0] and shape[1] of grads.";
    }
    std::vector<int64_t> output_shape{grad_shape[0], grad_shape[1], sizes[2], sizes[3]};
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    ShapeVector output_shape{grad_shape[0], grad_shape[1], abstract::Shape::kShapeDimAny,
                             abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr ResizeV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> x_nearest_valid_type = {kInt8, kUInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> x_linear_cubic_valid_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> roi_valid_types = {kFloat32};
  const std::set<TypePtr> scales_valid_types = {kFloat32};
  const std::set<TypePtr> sizes_valid_types = {kInt64, kInt32};
  TypePtr x_type = input_args[kInputIndex0]->BuildType();
  TypePtr roi_type = input_args[kInputIndex1]->BuildType();
  TypePtr scales_type = input_args[kInputIndex2]->BuildType();
  TypePtr sizes_type = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("roi", roi_type, roi_valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTypeValid("scales", scales_type, scales_valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTypeValid("original_size", sizes_type, sizes_valid_types, primitive->name());

  auto mode_ptr = primitive->GetAttr("mode");
  std::string mode_str = GetValue<std::string>(mode_ptr);

  if (mode_str == "nearest") {
    (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, x_nearest_valid_type, primitive->name());
  } else {
    (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, x_linear_cubic_valid_types, primitive->name());
  }
  return x_type;
}
}  // namespace

void ResizeV2Grad::set_coordinate_transformation_mode(const std::string coordinate_transformation_mode) {
  (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode));
}

std::string ResizeV2Grad::get_coordinate_transformation_mode() const {
  auto value_ptr = GetAttr("coordinate_transformation_mode");
  return GetValue<std::string>(value_ptr);
}

void ResizeV2Grad::set_mode(const std::string mode) { (void)this->AddAttr("mode", api::MakeValue(mode)); }

std::string ResizeV2Grad::get_mode() const {
  auto value_ptr = GetAttr("mode");
  return GetValue<std::string>(value_ptr);
}

void ResizeV2Grad::Init(const std::string coordinate_transformation_mode, const std::string mode) {
  this->set_coordinate_transformation_mode(coordinate_transformation_mode);
  this->set_mode(mode);
}

AbstractBasePtr ResizeV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ResizeV2GradInferType(primitive, input_args);
  auto infer_shape = ResizeV2GradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGResizeV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeV2GradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {3}; }
};

MIND_API_OPERATOR_IMPL(ResizeV2Grad, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeV2Grad, prim::kPrimResizeV2Grad, AGResizeV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
