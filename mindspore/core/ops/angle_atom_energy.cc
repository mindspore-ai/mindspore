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

#include "ops/angle_atom_energy.h"

#include <set>
#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AngleAtomEnergy, BaseOperator);
class AngleAtomEnergyInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kInputNum,
                                             prim_name);
    auto uint_crd_f_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto uint_crd_f_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(uint_crd_f_shape_ptr)[kShape];
    if (!IsDynamic(uint_crd_f_shape)) {
      (void)CheckAndConvertUtils::CheckInteger("uint_crd_f_shape", SizeToLong(uint_crd_f_shape.size()), kEqual, kTwo,
                                               prim_name);
      (void)CheckAndConvertUtils::CheckInteger("uint_crd_f_shape[1]", SizeToLong(uint_crd_f_shape[1]), kEqual, kThree,
                                               prim_name);
    }
    auto scaler_f_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto scaler_f_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(scaler_f_shape_ptr)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("scaler_f_shape", SizeToLong(scaler_f_shape.size()), kEqual, 1, prim_name);
    if (!IsDynamic(scaler_f_shape)) {
      (void)CheckAndConvertUtils::CheckInteger("scaler_f_shape", SizeToLong(scaler_f_shape[0]), kEqual, kThree,
                                               prim_name);
    }
    auto angle_numbers = GetValue<int64_t>(primitive->GetAttr("angle_numbers"));
    for (size_t input_index = 2; input_index < kInputNum; ++input_index) {
      auto cur_input_shape_ptr = input_args[input_index]->BuildShape();
      auto cur_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(cur_input_shape_ptr)[kShape];
      (void)CheckAndConvertUtils::CheckInteger("input_dim", SizeToLong(cur_input_shape.size()), kEqual, 1, prim_name);
      if (!IsDynamic(cur_input_shape)) {
        (void)CheckAndConvertUtils::CheckInteger("input_shape", SizeToLong(cur_input_shape[0]), kEqual, angle_numbers,
                                                 prim_name);
      }
    }
    ShapeVector out_shape{uint_crd_f_shape[0]};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    std::set<TypePtr> uint32_type = {kUInt32};
    auto uint_crd_f_dtype = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("uint_crd_f", uint_crd_f_dtype, uint32_type, prim_name);
    std::set<TypePtr> float32_type = {kFloat32};
    auto scaler_f_type = input_args[kInputIndex1]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("scaler_f", scaler_f_type, float32_type, prim_name);
    std::set<TypePtr> int32_type = {kInt32};
    auto atom_a_type = input_args[kInputIndex2]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("atom_a", atom_a_type, int32_type, prim_name);
    auto atom_b_type = input_args[kInputIndex3]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("atom_b", atom_b_type, int32_type, prim_name);
    auto atom_c_type = input_args[kInputIndex4]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("atom_c", atom_c_type, int32_type, prim_name);
    auto angle_k_type = input_args[kInputIndex5]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("angle_k", angle_k_type, float32_type, prim_name);
    auto angle_theta0_type = input_args[kInputIndex6]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("angle_theta0", angle_theta0_type, float32_type, prim_name);
    return angle_k_type;
  }

 private:
  static constexpr size_t kInputNum = 7;
  static constexpr size_t kTwo = 2;
  static constexpr size_t kThree = 3;
};

void AngleAtomEnergy::Init(const int64_t angle_numbers) { this->set_angle_numbers(angle_numbers); }

void AngleAtomEnergy::set_angle_numbers(const int64_t angle_numbers) {
  (void)this->AddAttr("angle_numbers", api::MakeValue(angle_numbers));
}

int64_t AngleAtomEnergy::get_angle_numbers() const {
  auto value_ptr = GetAttr("angle_numbers");
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(AngleAtomEnergy, prim::kPrimAngleAtomEnergy, AngleAtomEnergyInfer, false);
}  // namespace ops
}  // namespace mindspore
