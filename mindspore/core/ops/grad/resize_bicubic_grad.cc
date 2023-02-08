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
#include "ops/grad/resize_bicubic_grad.h"
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
constexpr size_t index3 = 3;
constexpr size_t num4 = 4;
abstract::ShapePtr ResizeBicubicGradInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto original_image_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (grads_shape.size() != num4) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The rank of grads shape need to be equal to 4, but got " << grads_shape.size();
  }
  if (original_image_shape.size() != num4) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the rank of original_image shape need to be equal to 4, but got "
                             << original_image_shape.size();
  }
  if (grads_shape[0] != original_image_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[0] is " << grads_shape[0]
                             << ", but the shape of original_image_shape[0] is " << original_image_shape[0]
                             << ". The first dimension of the shape of grads_shape "
                             << "must be equal to that of original_image_shape.";
  }
  if (grads_shape[index3] != original_image_shape[index3]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[3] is "
                             << grads_shape[index3] << ", but the shape of original_image_shape[3] is "
                             << original_image_shape[index3] << ". The third dimension of the shape of grads_shape "
                             << "must be equal to that of original_image_shape.";
  }
  return std::make_shared<abstract::Shape>(original_image_shape);
}

TypePtr ResizeBicubicGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  const std::set<TypePtr> valid0_types = {kFloat32, kFloat64};
  const std::set<TypePtr> valid1_types = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads_type", input_args[0]->BuildType(), valid0_types,
                                                   primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("original_image_type", input_args[1]->BuildType(), valid1_types,
                                                   primitive->name());
  return input_args[1]->BuildType();
}
}  // namespace
MIND_API_OPERATOR_IMPL(ResizeBicubicGrad, BaseOperator);
void ResizeBicubicGrad::set_align_corners(const bool align_corners) {
  (void)this->AddAttr("align_corners", api::MakeValue(align_corners));
}
void ResizeBicubicGrad::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers));
}

bool ResizeBicubicGrad::get_align_corners() const {
  auto value_ptr = GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}
bool ResizeBicubicGrad::get_half_pixel_centers() const {
  auto value_ptr = GetAttr("half_pixel_centers");
  return GetValue<bool>(value_ptr);
}

void ResizeBicubicGrad::Init(const bool align_corners, const bool half_pixel_centers) {
  this->set_align_corners(align_corners);
  this->set_half_pixel_centers(half_pixel_centers);
}

AbstractBasePtr ResizeBicubicGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ResizeBicubicGradInferType(primitive, input_args);
  auto infer_shape = ResizeBicubicGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ResizeBicubicGrad, prim::kPrimResizeBicubicGrad, ResizeBicubicGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
