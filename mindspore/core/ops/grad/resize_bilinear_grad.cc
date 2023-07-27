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
#include "ops/grad/resize_bilinear_grad.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/image_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
void ResizeBilinearGrad::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

bool ResizeBilinearGrad::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

void ResizeBilinearGrad::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr(kHalfPixelCenters, api::MakeValue(half_pixel_centers));
}

bool ResizeBilinearGrad::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::ShapePtr ResizeBilinearGradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  const int64_t kRank = 4;
  auto x_shape_val =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kOriginalImageIndex]->BuildShape())[kShape];
  if (!IsDynamicRank(x_shape_val)) {
    int64_t x_rank = SizeToLong(x_shape_val.size());
    if (x_rank != kRank) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x should have rank 4, but got " << x_rank << ".";
    }
  }

  auto dy_shape_val = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGradIndex]->BuildShape())[kShape];
  if (!IsDynamicRank(dy_shape_val)) {
    int64_t dy_rank = SizeToLong(dy_shape_val.size());
    if (dy_rank != kRank) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', dy should have rank 4, but got " << dy_rank << ".";
    }
  }

  return std::make_shared<abstract::Shape>(x_shape_val);
}

TypePtr ResizeBilinearGradInferType(const PrimitivePtr &primitive,
                                    const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_type = input_args[kOriginalImageIndex]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                            << ".";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[kInputIndex0]->BuildType(), valid_types,
                                                   prim_name);
  return x_type;
}
}  // namespace

void ResizeBilinearGrad::Init(const bool align_corners, const bool half_pixel_centers) {
  this->set_align_corners(align_corners);
  this->set_half_pixel_centers(half_pixel_centers);
}

MIND_API_OPERATOR_IMPL(ResizeBilinearGrad, BaseOperator);

AbstractBasePtr ResizeBilinearGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto infer_type = ResizeBilinearGradInferType(primitive, input_args);
  auto infer_shape = ResizeBilinearGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGResizeBilinearGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBilinearGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBilinearGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeBilinearGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeBilinearGrad, prim::kPrimResizeBilinearGrad, AGResizeBilinearGradInfer, false);
}  // namespace ops
}  // namespace mindspore
