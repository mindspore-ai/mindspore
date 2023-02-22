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

#include "ops/grad/affine_grid_grad.h"

#include <string>
#include <set>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int kLenImageSize2D = 4;
constexpr int kLenImageSize3D = 5;

abstract::ShapePtr AffineGridGradInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto y_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];

  auto y_grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  if (y_grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto x_size_arg = input_args[kInputIndex1];
  auto x_size_value_ptr = x_size_arg->BuildValue();
  if ((x_size_arg->isa<abstract::AbstractTuple>() && x_size_value_ptr->isa<ValueTuple>()) ||
      (x_size_arg->isa<abstract::AbstractTensor>() && x_size_value_ptr->isa<tensor::Tensor>())) {
    ShapeVector x_size_val;
    if (x_size_value_ptr->isa<ValueTuple>()) {
      x_size_val = CheckAndConvertUtils::CheckTupleInt("input[x_size]", x_size_value_ptr, prim_name);
    } else if (x_size_value_ptr->isa<tensor::Tensor>()) {  // 2-rd infer will be a tensor
      x_size_val = CheckAndConvertUtils::CheckTensorIntValue("x_size", x_size_value_ptr, prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                              << "the input[x_size] must be a tuple of int.";
    }
    (void)CheckAndConvertUtils::CheckPositiveVector("x_size", x_size_val, prim_name);
    int64_t x_size_val_size = SizeToLong(x_size_val.size());
    CheckAndConvertUtils::CheckInRange<int64_t>("size of 'x_size'", x_size_val_size, kIncludeBoth,
                                                {kLenImageSize2D, kLenImageSize3D}, prim_name);
    if (x_size_val[kInputIndex0] != y_grad_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                               << "the x_size[0] must be equal to the shape[0] of y_grad, "
                               << "but got the x_size[0] is " << x_size_val[kInputIndex0]
                               << " and the shape[0] of y_grad is " << y_grad_shape[kInputIndex0] << ".";
    }
    auto y_grad_rank = SizeToLong(y_grad_shape.size());
    ShapeVector x_grad_shape;
    if (x_size_val_size == kLenImageSize2D) {
      (void)CheckAndConvertUtils::CheckInteger("rank of 'y_grad'", y_grad_rank, kEqual, kLenImageSize2D, prim_name);
      if (y_grad_shape[kInputIndex1] == x_size_val[kInputIndex2] &&
          y_grad_shape[kInputIndex2] == x_size_val[kInputIndex3] && y_grad_shape[kInputIndex3] == kInputIndex2) {
        auto N = static_cast<int64_t>(x_size_val[kInputIndex0]);
        x_grad_shape = {N, kInputIndex2, kInputIndex3};
      } else {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                                 << "the shape of 'y_grad' must be [N, H, W, 2] and "
                                 << "the shape of 'x_size' must be [N, C, H, W] for 2D; "
                                 << "But got the shape of 'y_grad is [" << y_grad_shape[kInputIndex0] << ", "
                                 << y_grad_shape[kInputIndex1] << ", " << y_grad_shape[kInputIndex2] << ", "
                                 << y_grad_shape[kInputIndex3] << "] "
                                 << "and the size of 'x_size' is [" << x_size_val[kInputIndex0] << ", "
                                 << x_size_val[kInputIndex1] << ", " << x_size_val[kInputIndex2] << ", "
                                 << x_size_val[kInputIndex3] << "] ";
      }
    } else if (x_size_val_size == kLenImageSize3D) {
      (void)CheckAndConvertUtils::CheckInteger("rank of 'y_grad'", y_grad_rank, kEqual, kLenImageSize3D, prim_name);
      if (y_grad_shape[kInputIndex1] == x_size_val[kInputIndex2] &&
          y_grad_shape[kInputIndex2] == x_size_val[kInputIndex3] &&
          y_grad_shape[kInputIndex3] == x_size_val[kInputIndex4] && y_grad_shape[kInputIndex4] == kInputIndex3) {
        auto N = static_cast<int64_t>(x_size_val[kInputIndex0]);
        x_grad_shape = {N, kInputIndex3, kInputIndex4};
      } else {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                                 << "the shape of 'y_grad' must be [N, D, H, W, 3] and "
                                 << "the shape of 'x_size' must be [N, C, D, H, W] for 3D; "
                                 << "But got the shape of 'y_grad is [" << y_grad_shape[kInputIndex0] << ", "
                                 << y_grad_shape[kInputIndex1] << ", " << y_grad_shape[kInputIndex2] << ", "
                                 << y_grad_shape[kInputIndex3] << ", " << y_grad_shape[kInputIndex4] << "] "
                                 << "and the size of 'x_size' is [" << x_size_val[kInputIndex0] << ", "
                                 << x_size_val[kInputIndex1] << ", " << x_size_val[kInputIndex2] << ", "
                                 << x_size_val[kInputIndex3] << ", " << x_size_val[kInputIndex4] << "] ";
      }
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                               << "the size of 'x_size' must be 4 for 2D or 5 for 3D. ";
    }
    return std::make_shared<abstract::Shape>(x_grad_shape);
  } else {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
}

TypePtr AffineGridGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto y_grad_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(y_grad_type);
  const std::set<TypePtr> y_grad_valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y_grad", y_grad_type, y_grad_valid_types, op_name);
  auto x_size_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_size_type);
  const std::set<TypePtr> x_size_valid_types = {kTensorType, kTuple};  // 2-rd infer will be a tensor.
  (void)CheckAndConvertUtils::CheckTypeValid("x_size", x_size_type, x_size_valid_types, op_name);
  return y_grad_type;
}
}  // namespace

AbstractBasePtr AffineGridGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = AffineGridGradInferType(primitive, input_args);
  auto infer_shape = AffineGridGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void AffineGridGrad::Init(const bool align_corners) { set_align_corners(align_corners); }

void AffineGridGrad::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

bool AffineGridGrad::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(AffineGridGrad, BaseOperator);

// AG means auto generated
class MIND_API AGAffineGridGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AffineGridGrad, prim::kPrimAffineGridGrad, AGAffineGridGradInfer, false);
}  // namespace ops
}  // namespace mindspore
