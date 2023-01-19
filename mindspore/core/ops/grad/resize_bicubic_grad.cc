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

#include <algorithm>
#include <memory>
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
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t num4 = 4;
abstract::ShapePtr ResizeBicubicGradInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto original_image_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];

  std::vector<std::vector<int64_t>> all_shapes = {grads_shape, original_image_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);

  if (!is_dynamic_rank) {
    (void)CheckAndConvertUtils::CheckInteger("grads rank", grads_shape.size(), kEqual, num4, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("original image rank", original_image_shape.size(), kEqual, num4,
                                             prim_name);
  }
  if (!is_dynamic) {
    if (grads_shape[kInputIndex0] != original_image_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[0] is "
                               << grads_shape[kInputIndex0] << ", but the shape of original_image_shape[0] is "
                               << original_image_shape[kInputIndex0]
                               << ". The batch dimension of the shape of grads_shape "
                               << "must be equal to that of original_image_shape.";
    }
    if (grads_shape[kInputIndex1] != original_image_shape[kInputIndex1]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[1] is "
                               << grads_shape[kInputIndex1] << ", but the shape of original_image_shape[1] is "
                               << original_image_shape[kInputIndex1]
                               << ". The channel dimension of the shape of grads_shape "
                               << "must be equal to that of original_image_shape.";
    }
  }

  return std::make_shared<abstract::Shape>(original_image_shape);
}

TypePtr ResizeBicubicGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto grads_type = input_args[0]->BuildType();
  auto original_image_type = input_args[1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  const std::map<std::string, TypePtr> types = {{"grads_type", grads_type},
                                                {"original_image_type", original_image_type}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return grads_type;
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

// AG means auto generated
class MIND_API AGResizeBicubicGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBicubicGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBicubicGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBicubicGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeBicubicGrad, prim::kPrimResizeBicubicGrad, AGResizeBicubicGradInfer, false);
}  // namespace ops
}  // namespace mindspore
