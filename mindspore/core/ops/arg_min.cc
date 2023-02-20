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

#include <set>
#include <algorithm>

#include "ops/arg_min.h"
#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
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
#include "ir/scalar.h"
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
void ArgMin::Init(const int64_t axis, const TypeId output_type) {
  set_axis(axis);
  set_output_type(output_type);
}

void ArgMin::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
void ArgMin::set_output_type(const TypeId output_type) {
  (void)this->AddAttr(kOutputType, api::Type::GetType(output_type));
}

int64_t ArgMin::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

TypeId ArgMin::get_output_type() const {
  auto type_ptr = GetAttr(kOutputType)->cast<api::TensorTypePtr>()->element();
  return type_ptr->type_id();
}

void InferImplReduceFuncCalShape(const ShapeVector &x_shape, const int64_t axis_value, ShapeVector *shape) {
  (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
  (void)shape->erase(shape->begin() + axis_value);
}

int64_t InferImplArgMinFuncCheckAxis(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto axis_ptr = prim->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_ptr);
  if (!axis_ptr->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "For '" << prim->name() << "', 'axis' must be int.";
  }
  int64_t data_axis = GetValue<int64_t>(axis_ptr);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Argmin", input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto dim = input_shape.size();

  int64_t data_dim = static_cast<int64_t>(dim);
  if (data_axis < -data_dim || data_axis >= data_dim) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', 'axis' must be in [" << -data_dim << ", " << data_dim
                             << "). But got 'axis' = " << data_axis << ".";
  }
  int64_t ret_axis = data_axis;
  if (data_axis >= -data_dim && data_axis < 0) {
    ret_axis += data_dim;
  }
  return ret_axis;
}

abstract::ShapePtr ArgMinInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Argmin", input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  ShapeVector out_shape = {};
  int64_t axis_value = InferImplArgMinFuncCheckAxis(primitive, input_args);
  InferImplReduceFuncCalShape(input_shape, axis_value, &out_shape);
  return std::make_shared<abstract::Shape>(out_shape);
}

TensorTypePtr ArgMinInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kUInt32,
                                         kUInt64,  kInt8,    kInt16,   kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  const std::set<TypePtr> out_valid_types = {kInt32, kInt64};
  ValuePtr out_type_value = prim->GetAttr(kOutputType);
  TypePtr out_type_ptr = dyn_cast<Type>(out_type_value);
  (void)CheckAndConvertUtils::CheckTypeValid("output_type", out_type_ptr, out_valid_types, prim->name());
  return std::make_shared<TensorType>(out_type_ptr);
}

MIND_API_OPERATOR_NAME_IMPL(ArgMin, kNameArgMin, BaseOperator);
abstract::AbstractBasePtr ArgMinInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  auto type = ArgMinInferType(primitive, input_args);
  auto shape = ArgMinInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

using Argmin = ArgMin;

// AG means auto generated
class MIND_API AGArgMinInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMinInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMinInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMinInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Argmin, prim::kPrimArgmin, AGArgMinInfer, false);
}  // namespace ops
}  // namespace mindspore
