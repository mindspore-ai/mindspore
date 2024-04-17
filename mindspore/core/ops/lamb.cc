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

#include "ops/lamb.h"

#include <map>
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
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr LambInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[kInputIndex0]->GetType();
  auto m_type = input_args[kInputIndex1]->GetType();
  auto v_type = input_args[kInputIndex2]->GetType();
  auto lr_type = input_args[kInputIndex3]->GetType();
  auto beta1_type = input_args[kInputIndex4]->GetType();
  auto beta2_type = input_args[kInputIndex5]->GetType();
  auto epsilon_type = input_args[kInputIndex6]->GetType();
  auto decay_type = input_args[kInputIndex7]->GetType();
  auto global_step_type = input_args[kInputIndex8]->GetType();
  auto gradient_type = input_args[kInputIndex9]->GetType();

  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("var", var_type);
  (void)type_dict.emplace("m", m_type);
  (void)type_dict.emplace("v", v_type);
  std::set<TypePtr> num_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(type_dict, num_type, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", gradient_type, num_type, prim_name);
  std::map<std::string, TypePtr> type_dict1;
  (void)type_dict1.emplace("beta1", beta1_type);
  (void)type_dict1.emplace("beta2", beta2_type);
  (void)type_dict1.emplace("epsilon", epsilon_type);
  (void)type_dict1.emplace("lr", lr_type);
  (void)type_dict1.emplace("decay", decay_type);
  std::set<TypePtr> float_set = {kFloat32};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(type_dict1, float_set, prim_name, true);

  (void)CheckAndConvertUtils::CheckTypeValid("global_step", global_step_type, num_type, prim_name);

  return var_type;
}
BaseShapePtr LambInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto m_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto v_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto grad_shape_ptr = input_args[kInputIndex9]->GetShape();
  MS_EXCEPTION_IF_NULL(var_shape_ptr);
  MS_EXCEPTION_IF_NULL(m_shape_ptr);
  MS_EXCEPTION_IF_NULL(v_shape_ptr);
  MS_EXCEPTION_IF_NULL(grad_shape_ptr);
  if (var_shape_ptr->IsDynamic() || m_shape_ptr->IsDynamic() || v_shape_ptr->IsDynamic() ||
      grad_shape_ptr->IsDynamic()) {
    MS_LOG(WARNING) << "var is dynamic" << var_shape_ptr->IsDynamic() << "m is dynamic" << m_shape_ptr->IsDynamic()
                    << "v is dynamic" << v_shape_ptr->IsDynamic() << "grad is dynamic" << grad_shape_ptr->IsDynamic();
    return var_shape_ptr;
  }
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->GetShape())[kShape];
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, m_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, v_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, grad_shape, prim_name);

  return var_shape_ptr;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Lamb, BaseOperator);

AbstractBasePtr LambInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t kInputNum = 10;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = LambInferType(primitive, input_args);
  auto infer_shape = LambInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLambInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LambInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LambInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LambInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Lamb, prim::kPrimLamb, AGLambInfer, false);
}  // namespace ops
}  // namespace mindspore
