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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/grad/resize_linear_1d_grad.h"
#include "ops/image_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ResizeLinear1DGradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(grad_shape_ptr);
  auto grad_shape = grad_shape_ptr->shape();
  auto input_x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 1);
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  auto input_x_shape = input_x_shape_ptr->shape();
  std::vector<int64_t> ret_shape(kInputIndex3, abstract::Shape::kShapeDimAny);
  if (!IsDynamicRank(grad_shape)) {
    ret_shape[kInputIndex0] = grad_shape[kInputIndex0];
    ret_shape[kInputIndex1] = grad_shape[kInputIndex1];
  }
  if (!IsDynamicRank(input_x_shape)) {
    ret_shape[kInputIndex2] = input_x_shape[kInputIndex2];
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr ResizeLinear1DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args.at(kInputIndex1));
  auto x_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
}
}  // namespace

void ResizeLinear1DGrad::set_coordinate_transformation_mode(const std::string coordinate_transformation_mode) {
  (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode));
}
std::string ResizeLinear1DGrad::get_coordinate_transformation_mode() const {
  auto value_ptr = GetAttr("coordinate_transformation_mode");
  return GetValue<std::string>(value_ptr);
}

void ResizeLinear1DGrad::Init(const std::string coordinate_transformation_mode) {
  this->set_coordinate_transformation_mode(coordinate_transformation_mode);
}

MIND_API_OPERATOR_IMPL(ResizeLinear1DGrad, BaseOperator);
AbstractBasePtr ResizeLinear1DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  return abstract::MakeAbstract(ResizeLinear1DGradInferShape(primitive, input_args),
                                ResizeLinear1DGradInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGResizeLinear1DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeLinear1DGrad, prim::kPrimResizeLinear1DGrad, AGResizeLinear1DGradInfer, false);
}  // namespace ops
}  // namespace mindspore
