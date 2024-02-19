/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/silent_check_v2.h"

#include <map>
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
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SilentCheckV2InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  auto pre_val_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto min_val_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto max_val_shape_ptr = input_args[kInputIndex3]->BuildShape();
  auto result_shape_ptr = input_args[kInputIndex5]->BuildShape();
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{pre_val_shape_ptr, min_val_shape_ptr, max_val_shape_ptr, result_shape_ptr});
}

TuplePtr SilentCheckV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto pre_val_type_ptr = input_args[kInputIndex1]->BuildType();
  auto min_val_type_ptr = input_args[kInputIndex2]->BuildType();
  auto max_val_type_ptr = input_args[kInputIndex3]->BuildType();
  auto result_type_ptr = input_args[kInputIndex5]->BuildType();

  (void)CheckAndConvertUtils::CheckTypeValid("result", result_type_ptr, {kInt32}, prim_name);
  const std::map<std::string, TypePtr> types = {
    {"pre_val", pre_val_type_ptr},
    {"min_val", min_val_type_ptr},
    {"max_val", max_val_type_ptr},
  };
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, prim_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{pre_val_type_ptr, min_val_type_ptr, max_val_type_ptr, result_type_ptr});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SilentCheckV2, BaseOperator);
void SilentCheckV2::Init(const int64_t c_min_steps, const float c_thresh_l1, const float c_coeff_l1,
                         const float c_thresh_l2, const float c_coeff_l2) {
  this->set_c_min_steps(c_min_steps);
  this->set_c_thresh_l1(c_thresh_l1);
  this->set_c_coeff_l1(c_coeff_l1);
  this->set_c_thresh_l2(c_thresh_l2);
  this->set_c_coeff_l2(c_coeff_l2);
}

void SilentCheckV2::set_c_min_steps(const int64_t c_min_steps) {
  (void)this->AddAttr(kCMinSteps, api::MakeValue(c_min_steps));
}
int64_t SilentCheckV2::get_c_min_steps() const {
  auto value_ptr = GetAttr(kCMinSteps);
  return GetValue<int64_t>(value_ptr);
}
void SilentCheckV2::set_c_thresh_l1(const float c_thresh_l1) {
  (void)this->AddAttr(kCThreshL1, api::MakeValue(c_thresh_l1));
}
float SilentCheckV2::get_c_thresh_l1() const {
  auto value_ptr = GetAttr(kCThreshL1);
  return GetValue<float>(value_ptr);
}
void SilentCheckV2::set_c_coeff_l1(const float c_coeff_l1) {
  (void)this->AddAttr(kCCoeffL1, api::MakeValue(c_coeff_l1));
}
float SilentCheckV2::get_c_coeff_l1() const {
  auto value_ptr = GetAttr(kCCoeffL1);
  return GetValue<float>(value_ptr);
}
void SilentCheckV2::set_c_thresh_l2(const float c_thresh_l2) {
  (void)this->AddAttr(kCThreshL2, api::MakeValue(c_thresh_l2));
}
float SilentCheckV2::get_c_thresh_l2() const {
  auto value_ptr = GetAttr(kCThreshL2);
  return GetValue<float>(value_ptr);
}
void SilentCheckV2::set_c_coeff_l2(const float c_coeff_l2) {
  (void)this->AddAttr(kCCoeffL2, api::MakeValue(c_coeff_l2));
}
float SilentCheckV2::get_c_coeff_l2() const {
  auto value_ptr = GetAttr(kCCoeffL2);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr SilentCheckV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto type = SilentCheckV2InferType(primitive, input_args);
  auto shape = SilentCheckV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGSilentCheckV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SilentCheckV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SilentCheckV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SilentCheckV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SilentCheckV2, prim::kPrimSilentCheckV2, AGSilentCheckV2Infer, false);
}  // namespace ops
}  // namespace mindspore
