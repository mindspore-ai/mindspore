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
#include "ops/resize_linear_1d.h"
#include <algorithm>
#include <set>
#include <string>
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
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/image_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kResizeLinear1InputNum = 2;
constexpr int64_t kInputShape0Dim = 3;
constexpr int64_t kInputShape1Dim = 1;
abstract::ShapePtr ResizeLinear1DInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kResizeLinear1InputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  const int64_t expect_x_rank = 3;
  std::vector<int64_t> output_shape(expect_x_rank, abstract::Shape::kShapeDimAny);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("images' rank", SizeToLong(x_shape.size()), kEqual, expect_x_rank,
                                             prim_name);
    output_shape[kInputIndex0] = x_shape[kInputIndex0];
    output_shape[kInputIndex1] = x_shape[kInputIndex1];
  }

  auto size_value = GetShapeValue(primitive, input_args[kInputIndex1]);
  if (IsDynamicRank(size_value)) {
    size_value = ShapeVector{abstract::Shape::kShapeDimAny};
  }
  const int64_t size_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("size", SizeToLong(size_value.size()), kEqual, size_num, prim_name);
  if (!IsDynamic(size_value)) {
    const int64_t kNumZero = 0;
    for (size_t i = 0; i < size_value.size(); ++i) {
      (void)CheckAndConvertUtils::CheckInteger("size", size_value[i], kGreaterThan, kNumZero, prim_name);
    }
  }
  output_shape[kInputIndex2] = size_value[kInputIndex0];
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeLinear1DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kResizeLinear1InputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_type = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> valid0_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", x_type, valid0_types, prim_name);

  auto size_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(size_type);
  if (size_type->isa<TensorType>()) {
    const std::set<TypePtr> valid1_types = {kInt32, kInt64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, valid1_types, prim_name);
  }

  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ResizeLinear1D, BaseOperator);

void ResizeLinear1D::set_coordinate_transformation_mode(const std::string coordinate_transformation_mode) {
  (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode));
}
std::string ResizeLinear1D::get_coordinate_transformation_mode() const {
  auto value_ptr = GetAttr("coordinate_transformation_mode");
  return GetValue<std::string>(value_ptr);
}

void ResizeLinear1D::Init(const std::string coordinate_transformation_mode) {
  this->set_coordinate_transformation_mode(coordinate_transformation_mode);
}

abstract::AbstractBasePtr ResizeLinear1DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto infer_type = ResizeLinear1DInferType(primitive, input_args);
  auto infer_shape = ResizeLinear1DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGResizeLinear1DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeLinear1D, prim::kPrimResizeLinear1D, AGResizeLinear1DInfer, false);
}  // namespace ops
}  // namespace mindspore
