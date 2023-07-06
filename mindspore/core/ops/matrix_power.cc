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

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kExponent = "n";
constexpr size_t kMatrixPowerInputMinRank = 2;
constexpr int64_t kLastSecond = -2;

TypePtr MatrixPowerInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  auto n_value = GetValue<int64_t>(prim->GetAttr(kExponent));
  auto elem_type = TypeIdToType(x_type->cast<TensorTypePtr>()->element()->type_id());
  if (n_value < 0 && (elem_type == kFloat16 || common_integral_types.count(elem_type) > 0)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", integral types are not supported for n < 0.";
  }
  return x_type;
}

abstract::ShapePtr MatrixPowerInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  (void)CheckAndConvertUtils::CheckInteger("x's rank", static_cast<int64_t>(x_shape.size()), kGreaterEqual,
                                           kMatrixPowerInputMinRank, prim_name);
  if (!IsDynamic(x_shape) && x_shape.back() != x_shape.end()[kLastSecond]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", dim[-1] and dim[-2] of x should be the same"
                             << ", but got dim[-1]: " << x_shape.back()
                             << " and dim[-2]: " << x_shape.end()[kLastSecond] << ".";
  }
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

void MatrixPower::Init(const int64_t exponent) { set_exponent(exponent); }

void MatrixPower::set_exponent(const int64_t exponent) { (void)this->AddAttr(kExponent, api::MakeValue(exponent)); }

int64_t MatrixPower::get_exponent() const {
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
