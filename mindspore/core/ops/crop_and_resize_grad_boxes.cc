/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/crop_and_resize_grad_boxes.h"

#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void CropAndResizeGradBoxes::Init(ResizeMethod method) { this->set_method(method); }

void CropAndResizeGradBoxes::set_method(ResizeMethod method) {
  auto swi = (int64_t)method;
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

ResizeMethod CropAndResizeGradBoxes::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

namespace {
constexpr size_t kInputNums = 4;
constexpr size_t kGrads = 0;
constexpr size_t kGradsShapeLen = 4;
constexpr size_t kHeight = 1;
constexpr size_t kWidth = 2;
constexpr size_t kDepth = 3;
constexpr size_t kImages = 1;
constexpr size_t kImageShapeLen = 4;
constexpr size_t kBoxes = 2;
constexpr size_t kBoxesShapeLen = 2;
constexpr size_t kCoordinateLen = 4;
constexpr size_t kBoxIndex = 3;
constexpr size_t kBoxIndShapeLen = 1;
abstract::ShapePtr CropAndResizeGradBoxesInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // Infer shape
  auto input_shape0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGrads]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("grads rank", SizeToLong(input_shape0.size()), kEqual, kGradsShapeLen,
                                           prim_name);
  auto input_shape1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kImages]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("images rank", SizeToLong(input_shape1.size()), kEqual, kImageShapeLen,
                                           prim_name);
  auto input_shape2 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBoxes]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("boxes rank", SizeToLong(input_shape2.size()), kEqual, kBoxesShapeLen,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("shape[1] of boxes", SizeToLong(input_shape2[1]), kEqual, kCoordinateLen,
                                           prim_name);
  auto input_shape3 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBoxIndex]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("box_index rank", SizeToLong(input_shape3.size()), kEqual, kBoxIndShapeLen,
                                           prim_name);
  if (!(input_shape1[kHeight] > 0 && input_shape1[kWidth] > 0)) {
    MS_EXCEPTION(ValueError) << "the height and width of images must be over 0 ";
  }
  if (!(input_shape0[kHeight] > 0 && input_shape0[kWidth] > 0)) {
    MS_EXCEPTION(ValueError) << "the height and width of grads must be over 0 ";
  }
  if (!(input_shape0[0] == input_shape3[0] && input_shape2[0] == input_shape0[0])) {
    MS_EXCEPTION(ValueError) << "the first dimension of the tensors in {grads, boxes, box_index} must be equal.";
  }
  if (input_shape0[kDepth] != input_shape1[kDepth]) {
    MS_EXCEPTION(ValueError) << "the depth of grads and images must be equal.";
  }
  return std::make_shared<abstract::Shape>(input_shape2);
}

TypePtr CropAndResizeGradBoxesInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNums, prim_name);
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[kGrads]->BuildType(), {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[kImages]->BuildType(), valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("boxes", input_args[kBoxes]->BuildType(), {kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("box_index", input_args[kBoxIndex]->BuildType(), {kInt32},
                                                   prim_name);
  return kFloat32;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CropAndResizeGradBoxes, BaseOperator);
AbstractBasePtr CropAndResizeGradBoxesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto type = CropAndResizeGradBoxesInferType(primitive, input_args);
  auto shape = CropAndResizeGradBoxesInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CropAndResizeGradBoxes, prim::kPrimCropAndResizeGradBoxes, CropAndResizeGradBoxesInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
