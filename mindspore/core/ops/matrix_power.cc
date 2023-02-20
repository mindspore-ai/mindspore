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
#include "ops/matrix_power.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kExponent = "n";

TypePtr MatrixPowerInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return x_type;
}

abstract::ShapePtr MatrixPowerInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const constexpr int64_t x_shape_size = 3;
  const constexpr int64_t x_shape_two = 2;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (x_shape.size() != x_shape_size) {
    MS_EXCEPTION(ValueError) << "For MatrixPower, x should be a 3-D tensor"
                             << ", but got x is a " << x_shape.size() << "-D tensor.";
  }
  if (x_shape[1] != x_shape[x_shape_two]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", sizes of dim[1] and dim[2] of x should be the same"
                             << ", but size of dim[1] of got x is " << x_shape[1] << ", size of dim[2] of got x is "
                             << x_shape[x_shape_two] << ".";
  }
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

void MatrixPower::Init(const int64_t exponent) { set_exponent(exponent); }

void MatrixPower::set_exponent(const int64_t exponent) { (void)this->AddAttr(kExponent, api::MakeValue(exponent)); }

int64_t MatrixPower::get_exponent() {
  auto value_ptr = GetAttr(kExponent);
  return GetValue<int64_t>(value_ptr);
}

MIND_API_OPERATOR_IMPL(MatrixPower, BaseOperator);
AbstractBasePtr MatrixPowerInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MatrixPowerInferType(primitive, input_args);
  auto infer_shape = MatrixPowerInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMatrixPowerInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixPowerInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixPowerInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixPowerInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixPower, prim::kPrimMatrixPower, AGMatrixPowerInfer, false);
}  // namespace ops
}  // namespace mindspore
