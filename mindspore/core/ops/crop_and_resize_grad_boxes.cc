/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
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
void CropAndResizeGradBoxes::Init(ResizeMethod method) { this->set_method(method); }

void CropAndResizeGradBoxes::set_method(ResizeMethod method) {
  auto swi = static_cast<int64_t>(method);
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

ResizeMethod CropAndResizeGradBoxes::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

namespace {
constexpr int64_t kInputNums = 4;
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
  MS_EXCEPTION_IF_NULL(input_args[kGrads]);
  auto input_shape0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGrads]->BuildShape())[kShape];
  MS_EXCEPTION_IF_NULL(input_args[kImages]);
  auto input_shape1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kImages]->BuildShape())[kShape];
  MS_EXCEPTION_IF_NULL(input_args[kBoxes]);
  auto input_shape2 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBoxes]->BuildShape())[kShape];
  MS_EXCEPTION_IF_NULL(input_args[kBoxIndex]);
  auto input_shape3 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBoxIndex]->BuildShape())[kShape];
  if (IsDynamicRank(input_shape0) || IsDynamicRank(input_shape1) || IsDynamicRank(input_shape2) ||
      IsDynamicRank(input_shape3)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  } else if (IsDynamic(input_shape0) || IsDynamic(input_shape1) || IsDynamic(input_shape2) || IsDynamic(input_shape3)) {
    return std::make_shared<abstract::Shape>(input_shape2);
  }

  (void)CheckAndConvertUtils::CheckInteger("grads rank", SizeToLong(input_shape0.size()), kEqual, kGradsShapeLen,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("images rank", SizeToLong(input_shape1.size()), kEqual, kImageShapeLen,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("boxes rank", SizeToLong(input_shape2.size()), kEqual, kBoxesShapeLen,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("shape[1] of boxes", input_shape2[1], kEqual, SizeToLong(kCoordinateLen),
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("box_index rank", SizeToLong(input_shape3.size()), kEqual, kBoxIndShapeLen,
                                           prim_name);
  if (!(input_shape1[kHeight] > 0 && input_shape1[kWidth] > 0)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the height and width of images must be greater than 0. But got height: "
                             << input_shape1[kHeight] << ", width: " << input_shape1[kWidth] << ".";
  }
  if (!(input_shape0[kHeight] > 0 && input_shape0[kWidth] > 0)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the height and width of grads must be greater than 0. But got height: "
                             << input_shape1[kHeight] << ", width: " << input_shape1[kWidth] << ".";
  }
  if (!(input_shape0[0] == input_shape3[0] && input_shape2[0] == input_shape0[0])) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', the first dimension of the tensors in {grads, boxes, box_index} must be equal. But got grads[0]: "
      << input_shape0[0] << ", boxes[0]: " << input_shape2[0] << ", box_index[0]: " << input_shape3[0] << ".";
  }
  if (input_shape0[kDepth] != input_shape1[kDepth]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the depth of grads and images must be equal. But grads depth: "
                             << input_shape0[kDepth] << ", images depth: " << input_shape1[kDepth] << ".";
  }
  return std::make_shared<abstract::Shape>(input_shape2);
}

TypePtr CropAndResizeGradBoxesInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNums, prim_name);
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_others = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[kGrads]->BuildType(), valid_others, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[kImages]->BuildType(), valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("boxes", input_args[kBoxes]->BuildType(), valid_others, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("box_index", input_args[kBoxIndex]->BuildType(), {kInt32},
                                                   prim_name);
  return input_args[kGrads]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(CropAndResizeGradBoxes, BaseOperator);
AbstractBasePtr CropAndResizeGradBoxesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto type = CropAndResizeGradBoxesInferType(primitive, input_args);
  auto shape = CropAndResizeGradBoxesInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGCropAndResizeGradBoxesInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CropAndResizeGradBoxesInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CropAndResizeGradBoxesInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CropAndResizeGradBoxesInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CropAndResizeGradBoxes, prim::kPrimCropAndResizeGradBoxes,
                                 AGCropAndResizeGradBoxesInfer, false);
}  // namespace ops
}  // namespace mindspore
