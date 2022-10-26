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
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void AttrTest(bool a, bool b) {
  if (a && b) {
    MS_EXCEPTION(ValueError) << "The half_pixel_centers must be false when align_corners is true "
                             << ", but half_pixel_centers got True";
  }
}

abstract::ShapePtr ResizeBicubicInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  const int64_t shape0_dim = 4;
  const int64_t shape1_dim = 1;
  constexpr int64_t indexid3 = 3;
  constexpr int64_t calnum2 = 2;
  constexpr int64_t calnum3 = 3;
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', images only support tensor!";
  }
  if (!input_args[1]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', size only support tensor!";
  }
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t kMaxLen = GetValue<int64_t>(max_length_ptr);
  auto input0_shape = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input0_shape);
  auto input0_shape_value_ptr = input0_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input0_shape_value_ptr);
  auto input0_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input0_type);
  auto input0_type_id = input0_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input0_type_id);
  auto input0_type_element = input0_type_id->element();
  MS_EXCEPTION_IF_NULL(input0_type_element);
  auto input1_shape = input_args[1]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input1_shape);
  auto input1_shape_value_ptr = input1_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input1_shape_value_ptr);
  auto input1_shape_tensor = input1_shape_value_ptr->cast<tensor::TensorPtr>();
  auto input1_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(input1_type);
  auto input1_type_id = input1_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input1_type_id);
  auto input1_type_element = input1_type_id->element();
  MS_EXCEPTION_IF_NULL(input1_type_element);
  auto shape0_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape]);
  auto shape1_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape]);
  auto shape0_v = shape0_ptr->shape();
  auto shape1_v = shape1_ptr->shape();
  if (shape0_v.size() != shape0_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the images tensor must be a 4-D tensor. But got "
                             << shape0_v.size() << "-D";
  }
  if (shape1_v.size() != shape1_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size tensor must be a 1-D tensor. But got "
                             << shape1_v.size() << "-D";
  }
  if (shape1_v[0] != calnum2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size shape must be 2. But got " << shape1_v[0];
  }
  auto align_corners_ptr = primitive->GetAttr("align_corners");
  bool align_corners = GetValue<bool>(align_corners_ptr);
  auto half_pixel_centers_ptr = primitive->GetAttr("half_pixel_centers");
  bool half_pixel_centers = GetValue<bool>(half_pixel_centers_ptr);
  AttrTest(align_corners, half_pixel_centers);
  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    auto input1_shape_ptr = static_cast<int32_t *>(input1_shape_tensor->data_c());
    if (input1_shape_ptr[0] <= 0 || input1_shape_ptr[1] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size must be positive "
                               << ", but got " << input1_shape_ptr[0] << " , " << input1_shape_ptr[1];
    }
    std::vector<int64_t> output_shape;
    auto shape_m = 1;
    output_shape.push_back(shape0_v[0]);
    output_shape.push_back(input1_shape_ptr[0]);
    output_shape.push_back(input1_shape_ptr[1]);
    output_shape.push_back(shape0_v[calnum3]);
    shape_m = shape0_v[0] * input1_shape_ptr[0] * input1_shape_ptr[1] * shape0_v[calnum3];
    if (shape_m > kMaxLen) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the number of elements of output must be less than max length: " << kMaxLen
                               << ", but got " << shape_m
                               << "! The shape of  output should be reduced or max_length should be increased";
    }
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    auto prim_name = primitive->name();
    auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
    auto x_shape = x_shape_ptr->shape();
    if (x_shape_ptr->IsDynamic()) {
      return std::make_shared<abstract::Shape>(x_shape);
    }
    ShapeVector shape_out = {shape0_v[0], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                             shape0_v[indexid3]};
    return std::make_shared<abstract::Shape>(shape_out);
  }
}
TypePtr ResizeBicubicInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid0_types = {kInt8, kUInt8, kInt16, kUInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid1_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[0]->BuildType(), valid0_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[1]->BuildType(), valid1_types, prim_name);
  string inputFp64 = "Float64";
  if (input_args[0]->BuildType()->ToString().find(inputFp64) != string::npos) {
    return kFloat64;
  }
  return kFloat32;
}
}  // namespace
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

AbstractBasePtr ResizeBicubicInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ResizeBicubicInferType(primitive, input_args);
  auto infer_shape = ResizeBicubicInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ResizeBicubic, prim::kPrimResizeBicubic, ResizeBicubicInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
