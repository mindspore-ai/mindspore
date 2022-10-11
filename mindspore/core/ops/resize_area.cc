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
#include "ops/resize_area.h"
#include <cmath>
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ResizeAreaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t size_num = 2;
  constexpr size_t indexid2 = 2;
  constexpr size_t indexid3 = 3;
  constexpr int64_t image_shape_size = 4;
  constexpr int64_t size_shape_size = 1;
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
  auto images_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("images dimension", SizeToLong(images_shape.size()), kEqual,
                                           image_shape_size, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("size dimension", SizeToLong(size_shape.size()), kEqual, size_shape_size,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("input1 num", size_shape[0], kEqual, size_num, primitive->name());

  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    auto input1_shape_ptr = static_cast<int32_t *>(input1_shape_tensor->data_c());
    if (input1_shape_ptr[0] <= 0 || input1_shape_ptr[1] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size must be positive "
                               << ", but got " << input1_shape_ptr[0] << " , " << input1_shape_ptr[1];
    }
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i <= indexid3; ++i) {
      if (i == 0 || i == indexid3) {
        output_shape.push_back(images_shape[i]);
      } else {
        output_shape.push_back(input1_shape_ptr[i - 1]);
      }
    }
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    auto prim_name = primitive->name();
    auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
    auto x_shape = x_shape_ptr->shape();
    if (x_shape_ptr->IsDynamic()) {
      auto x_min_shape = x_shape_ptr->min_shape();
      auto x_max_shape = x_shape_ptr->max_shape();
      x_min_shape[1] = 0;
      x_min_shape[indexid2] = 0;
      x_max_shape[1] = 1;
      x_max_shape[indexid2] = 1;
      return std::make_shared<abstract::Shape>(x_shape, x_min_shape, x_max_shape);
    }
    ShapeVector out_shape = {images_shape[0], abstract::Shape::SHP_ANY, abstract::Shape::SHP_ANY,
                             images_shape[indexid3]};
    ShapeVector shape_min = {images_shape[0], 0, 0, images_shape[indexid3]};
    ShapeVector shape_max = {images_shape[0], 1, 1, images_shape[indexid3]};
    return std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
  }
}
TypePtr ResizeAreaInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) { return kFloat32; }
}  // namespace
MIND_API_OPERATOR_IMPL(ResizeArea, BaseOperator);
void ResizeArea::Init(const bool align_corners) { this->set_align_corners(align_corners); }
void ResizeArea::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}
bool ResizeArea::get_align_corners() const { return GetValue<bool>(GetAttr(kAlignCorners)); }
AbstractBasePtr ResizeAreaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kInt8, kUInt8, kInt16, kUInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_types_1 = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[0]->BuildType(), valid_types,
                                                   primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[1]->BuildType(), valid_types_1,
                                                   primitive->name());
  auto infer_shape = ResizeAreaInferShape(primitive, input_args);
  auto infer_type = ResizeAreaInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ResizeArea, prim::kPrimResizeArea, ResizeAreaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
