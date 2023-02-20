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

#include "ops/affine_grid.h"

#include <memory>
#include <string>
#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
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
constexpr int RANK_THETA = 3;
constexpr int N_ROWS_THETA_4D = 2;
constexpr int N_COLS_THETA_4D = 3;
constexpr int LEN_IMAGE_SIZE_4D = 4;
constexpr int N_ROWS_THETA_5D = 3;
constexpr int N_COLS_THETA_5D = 4;
constexpr int LEN_IMAGE_SIZE_5D = 5;

abstract::ShapePtr AffineGridInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto theta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto theta_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  if (theta_shape_ptr->IsDynamic()) {
    // theta is dynamic shape, verification could not be performed.
    // launch kernel will fail, and infer shape will run again.
    ShapeVector grid_shape = {-2};
    return std::make_shared<abstract::Shape>(grid_shape);
  }
  auto theta_rank = SizeToLong(theta_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of 'theta'", theta_rank, kEqual, RANK_THETA, prim_name);
  auto output_size_arg = input_args[kInputIndex1];
  auto output_size_value_ptr = output_size_arg->BuildValue();
  if (IsValueKnown(output_size_value_ptr)) {
    ShapeVector output_size_val;
    if (output_size_value_ptr->isa<ValueTuple>()) {
      output_size_val = CheckAndConvertUtils::CheckTupleInt("input[output_size]", output_size_value_ptr, prim_name);
    } else if (output_size_value_ptr->isa<tensor::Tensor>()) {  // 2-rd infer will be a tensor
      output_size_val = CheckAndConvertUtils::CheckTensorIntValue("output_size", output_size_value_ptr, prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                              << "the input[output_size] must be a tuple of int.";
    }
    (void)CheckAndConvertUtils::CheckPositiveVector("output_size", output_size_val, prim_name);
    int64_t output_size_val_size = SizeToLong(output_size_val.size());
    CheckAndConvertUtils::CheckInRange<int64_t>("size of 'output_size'", output_size_val_size, kIncludeBoth,
                                                {LEN_IMAGE_SIZE_4D, LEN_IMAGE_SIZE_5D}, prim_name);
    if (output_size_val[kInputIndex0] != theta_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                               << "the output_size[0] must be equal to the shape[0] of theta, "
                               << "but got the output_size[0] is " << output_size_val[kInputIndex0]
                               << " and the shape[0] of theta is " << theta_shape[kInputIndex0] << ".";
    }
    ShapeVector grid_shape;
    if (output_size_val_size == LEN_IMAGE_SIZE_4D && theta_shape[kInputIndex1] == N_ROWS_THETA_4D &&
        theta_shape[kInputIndex2] == N_COLS_THETA_4D) {
      auto N = static_cast<int64_t>(output_size_val[kInputIndex0]);
      auto H = static_cast<int64_t>(output_size_val[kInputIndex2]);
      auto W = static_cast<int64_t>(output_size_val[kInputIndex3]);
      grid_shape = {N, H, W, kInputIndex2};
    } else if (output_size_val_size == LEN_IMAGE_SIZE_5D && theta_shape[kInputIndex1] == N_ROWS_THETA_5D &&
               theta_shape[kInputIndex2] == N_COLS_THETA_5D) {
      auto N = static_cast<int64_t>(output_size_val[kInputIndex0]);
      auto D = static_cast<int64_t>(output_size_val[kInputIndex2]);
      auto H = static_cast<int64_t>(output_size_val[kInputIndex3]);
      auto W = static_cast<int64_t>(output_size_val[kInputIndex4]);
      grid_shape = {N, D, H, W, kInputIndex3};
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                               << "the shape of 'theta' must be [N, 2, 3] and "
                               << "the size of 'output_size' must be 4 for 2D; "
                               << "or the shape of 'theta' must be [N, 3, 4] and "
                               << "the size of 'output_size' must be 5 for 3D. "
                               << "But got the shape of 'theta is [" << theta_shape[kInputIndex0] << ", "
                               << theta_shape[kInputIndex1] << ", " << theta_shape[kInputIndex2] << "] "
                               << "and the size of 'output_size' is " << output_size_val_size << ".";
    }
    return std::make_shared<abstract::Shape>(grid_shape);
  } else if (output_size_arg->isa<abstract::AbstractTensor>() || output_size_arg->isa<abstract::AbstractTuple>()) {
    ShapeVector grid_shape = {-2};
    return std::make_shared<abstract::Shape>(grid_shape);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                            << "the input[output_size] is not supported: " << output_size_arg->ToString();
  }
}

TypePtr AffineGridInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = prim->name();
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto theta_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(theta_type);
  const std::set<TypePtr> theta_valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("theta", theta_type, theta_valid_types, op_name);
  auto output_size_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(output_size_type);
  const std::set<TypePtr> output_size_valid_types = {kTensorType, kTuple};  // 2-rd infer will be a tensor.
  (void)CheckAndConvertUtils::CheckTypeValid("output_size", output_size_type, output_size_valid_types, op_name);
  return theta_type;
}
}  // namespace

void AffineGrid::Init(const bool align_corners) { set_align_corners(align_corners); }

bool AffineGrid::get_align_corners() const {
  auto value_ptr = this->GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

void AffineGrid::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

AbstractBasePtr AffineGridInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto align_corners_ptr = primitive->GetAttr(kAlignCorners);
  MS_EXCEPTION_IF_NULL(align_corners_ptr);
  auto infer_type = AffineGridInferType(primitive, input_args);
  auto infer_shape = AffineGridInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(AffineGrid, BaseOperator);

// AG means auto generated
class MIND_API AGAffineGridInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AffineGridInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AffineGrid, prim::kPrimAffineGrid, AGAffineGridInfer, false);
}  // namespace ops
}  // namespace mindspore
