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
#include "ops/fused_sparse_ftrl.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/overload.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace fused_sparse_ftrl {
// "var","accum","linear","grad","indices"
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kLinearIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kIndicesIndex = 4;
constexpr size_t kFusedSparseFtrlInputNum = 5;

abstract::TupleShapePtr FusedSparseFtrlInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  // the output is useless, so we don't have to focus on the output shape, cannot return 1
  auto var_shape_r = input_args[kVarIndex]->Broaden()->BuildShape();
  auto accum_shape_r = input_args[kAccumIndex]->Broaden()->BuildShape();
  auto linear_shape_r = input_args[kLinearIndex]->Broaden()->BuildShape();
  auto outputs = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>({var_shape_r, accum_shape_r, linear_shape_r}));
  for (auto &input : input_args) {
    if (input->BuildShape()->IsDynamic()) {
      return outputs;
    }
  }
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kVarIndex]->BuildShape())[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kAccumIndex]->BuildShape())[kShape];
  auto linear_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kLinearIndex]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndicesIndex]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGradIndex]->BuildShape())[kShape];

  (void)CheckAndConvertUtils::CheckValue("var shape", var_shape, kEqual, "accum shape", accum_shape, prim_name);
  (void)CheckAndConvertUtils::CheckValue("var shape", var_shape, kEqual, "linear shape", linear_shape, prim_name);
  // indices rank == 1
  (void)CheckAndConvertUtils::CheckInteger("indices rank", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  // grad_shape[0] == indices_shape[0]
  (void)CheckAndConvertUtils::CheckInteger("grad rank", SizeToLong(grad_shape.size()), kGreaterEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckValue("grad_shape[0]", grad_shape[0], kEqual, "indices_shape[0]", indices_shape[0],
                                         prim_name);
  // grad_shape[1:] == var_shape[1:] while grad_shape[0] == indices_shape[0]
  if (var_shape.size() > 1) {
    auto left_shape = var_shape;
    auto right_shape = grad_shape;
    (void)left_shape.erase(left_shape.begin());
    (void)right_shape.erase(right_shape.begin());
    (void)CheckAndConvertUtils::CheckValue("var_shape[1:]", left_shape, kEqual, "grad_shape[1:]", right_shape,
                                           prim_name);
  }
  return outputs;
}

TypePtr FusedSparseFtrlInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  std::map<std::string, TypePtr> types = {{"var", input_args[kVarIndex]->BuildType()},
                                          {"accum", input_args[kAccumIndex]->BuildType()},
                                          {"linear", input_args[kLinearIndex]->BuildType()},
                                          {"grad", input_args[kGradIndex]->BuildType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, prim_name);

  auto indices_dtype = input_args[kIndicesIndex]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_dtype, {kInt32}, prim_name);

  auto type = input_args[kVarIndex]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type});
}
}  // namespace fused_sparse_ftrl

void FusedSparseFtrl::set_lr(float lr) { (void)this->AddAttr(kLr, api::MakeValue(lr)); }

float FusedSparseFtrl::get_lr() const {
  auto value_ptr = GetAttr(kLr);
  return GetValue<float>(value_ptr);
}

void FusedSparseFtrl::set_l1(float l1) { (void)this->AddAttr(kL1, api::MakeValue(l1)); }

float FusedSparseFtrl::get_l1() const {
  auto value_ptr = GetAttr(kL1);
  return GetValue<float>(value_ptr);
}

void FusedSparseFtrl::set_l2(float l2) { (void)this->AddAttr(kL2, api::MakeValue(l2)); }

float FusedSparseFtrl::get_l2() const {
  auto value_ptr = GetAttr(kL2);
  return GetValue<float>(value_ptr);
}

void FusedSparseFtrl::set_lr_power(float lr_power) { (void)this->AddAttr(kLrPower, api::MakeValue(lr_power)); }

float FusedSparseFtrl::get_lr_power() const {
  auto value_ptr = GetAttr(kLrPower);
  return GetValue<float>(value_ptr);
}

void FusedSparseFtrl::set_use_locking(bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool FusedSparseFtrl::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void FusedSparseFtrl::Init(float lr, float l1, float l2, float lr_power, bool use_locking) {
  set_lr(lr);
  set_l1(l1);
  set_l2(l2);
  set_lr_power(lr_power);
  set_use_locking(use_locking);
}

MIND_API_OPERATOR_IMPL(FusedSparseFtrl, BaseOperator);
AbstractBasePtr FusedSparseFtrlInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  // float lr, float l1, float l2, float lr_power
  auto lr = GetValue<float>(primitive->GetAttr(kLr));
  auto l1 = GetValue<float>(primitive->GetAttr(kL1));
  auto l2 = GetValue<float>(primitive->GetAttr(kL2));
  auto lr_power = GetValue<float>(primitive->GetAttr(kLrPower));

  (void)CheckAndConvertUtils::CheckValue(kLr, lr, kGreaterThan, 0.0f, op_name);
  (void)CheckAndConvertUtils::CheckValue(kL1, l1, kGreaterEqual, 0.0f, op_name);
  (void)CheckAndConvertUtils::CheckValue(kL2, l2, kGreaterEqual, 0.0f, op_name);
  (void)CheckAndConvertUtils::CheckValue(kLrPower, lr_power, kLessEqual, 0.0f, op_name);

  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(fused_sparse_ftrl::kFusedSparseFtrlInputNum), op_name);
  auto types = fused_sparse_ftrl::FusedSparseFtrlInferType(primitive, input_args);
  auto shapes = fused_sparse_ftrl::FusedSparseFtrlInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGFusedSparseFtrlInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return fused_sparse_ftrl::FusedSparseFtrlInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return fused_sparse_ftrl::FusedSparseFtrlInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FusedSparseFtrlInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FusedSparseFtrl, prim::kPrimFusedSparseFtrl, AGFusedSparseFtrlInfer, false);
}  // namespace ops
}  // namespace mindspore
