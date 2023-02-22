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

#include <set>
#include <string>
#include <algorithm>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
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
const int64_t kInputShape0Dim = 3;
const int64_t kInputShape1Dim = 1;
abstract::ShapePtr ResizeLinear1DInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_arg = input_args[kInputIndex0];
  auto size_arg = input_args[kInputIndex1];
  if (!input_x_arg->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "Images only support tensor!";
  }
  if (!size_arg->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "Size only support tensor!";
  }
  auto input_x_shape = input_x_arg->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_x_shape);
  auto input_x_shape_value_ptr = input_x_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_x_shape_value_ptr);
  auto input_x_type = input_x_arg->BuildType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  auto input_x_type_id = input_x_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_x_type_id);
  auto input_x_type_element = input_x_type_id->element();
  MS_EXCEPTION_IF_NULL(input_x_type_element);
  auto size_shape = size_arg->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(size_shape);
  auto size_shape_value_ptr = size_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(size_shape_value_ptr);
  auto size_shape_tensor = size_shape_value_ptr->cast<tensor::TensorPtr>();
  auto size_type = size_arg->BuildType();
  MS_EXCEPTION_IF_NULL(size_type);
  auto size_type_id = size_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(size_type_id);
  auto size_type_element = size_type_id->element();
  MS_EXCEPTION_IF_NULL(size_type_element);
  auto shape0_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_arg->BuildShape())[kShape]);
  auto shape1_ptr =
    std::make_shared<abstract::Shape>(CheckAndConvertUtils::ConvertShapePtrToShapeMap(size_arg->BuildShape())[kShape]);
  auto shape0_v = shape0_ptr->shape();
  auto shape1_v = shape1_ptr->shape();
  // support dynamic shape
  if (IsDynamicRank(shape0_v) || IsDynamicRank(shape1_v)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  if (shape0_v.size() < kInputShape0Dim) {
    MS_EXCEPTION(ValueError) << "For 'ResizeLinear1D', the rank of images tensor must be greater than 3. But got "
                             << shape0_v.size();
  }
  if (shape1_v.size() != kInputShape1Dim) {
    MS_EXCEPTION(ValueError) << "For 'ResizeLinear1D', the size tensor must be a 1-D tensor. But got "
                             << shape1_v.size() << "-D";
  }
  if (size_arg->isa<abstract::AbstractTensor>() && size_arg->BuildValue()->isa<tensor::Tensor>()) {
    int64_t out_width = 0;
    if (size_shape_tensor->data_type() == kNumberTypeInt32) {
      auto size_shape_ptr = reinterpret_cast<int32_t *>(size_shape_tensor->data_c());
      out_width = static_cast<int64_t>(size_shape_ptr[kInputIndex0]);
    } else if (size_shape_tensor->data_type() == kNumberTypeInt64) {
      auto size_shape_ptr = reinterpret_cast<int64_t *>(size_shape_tensor->data_c());
      out_width = size_shape_ptr[kInputIndex0];
    }
    if (out_width <= 0) {
      MS_EXCEPTION(ValueError) << "The size must be positive , but got " << out_width;
    }

    std::vector<int64_t> output_shape = shape0_v;
    output_shape.pop_back();
    output_shape.push_back(out_width);
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    ShapeVector shape_out = shape0_v;
    shape_out.pop_back();
    shape_out.push_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(shape_out);
  }
}
TypePtr ResizeLinear1DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "For 'ResizeLinear1D', input args contain nullptr.";
  }
  auto prim_name = primitive->name();
  auto input_x_arg = input_args[kInputIndex0];
  auto size_arg = input_args[kInputIndex1];
  const std::set<TypePtr> valid0_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid1_types = {kInt64, kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_x_arg->BuildType(), valid0_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_arg->BuildType(), valid1_types, prim_name);
  return input_x_arg->BuildType();
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
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
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
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeLinear1D, prim::kPrimResizeLinear1D, AGResizeLinear1DInfer, false);
}  // namespace ops
}  // namespace mindspore
