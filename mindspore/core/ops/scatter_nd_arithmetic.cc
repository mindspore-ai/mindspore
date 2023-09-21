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

#include <map>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "ops/scatter_nd_add.h"
#include "ops/scatter_nd_div.h"
#include "ops/scatter_nd_max.h"
#include "ops/scatter_nd_min.h"
#include "ops/scatter_nd_mul.h"
#include "ops/scatter_nd_sub.h"
#include "ops/scatter_nd_update.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kIndexMinSize = 2;
constexpr auto kUpdateMinSize = 1;

abstract::ShapePtr ScatterNdArithmeticInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  auto updates_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(updates_shape_ptr);
  if (input_x_shape_ptr->IsDynamic() || indices_shape_ptr->IsDynamic() || updates_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];

  const int64_t input_x_size = SizeToLong(input_x_shape.size());
  const int64_t indices_size = SizeToLong(indices_shape.size());
  const int64_t updates_size = SizeToLong(updates_shape.size());

  (void)CheckAndConvertUtils::CheckValue<int64_t>("dimension of 'indices'", indices_size, kGreaterEqual, kIndexMinSize,
                                                  prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("dimension of 'updates'", updates_size, kGreaterEqual, kUpdateMinSize,
                                                  prim_name);

  const int64_t last_dim = indices_shape.back();
  (void)CheckAndConvertUtils::CheckValue("the value of last dimension of 'indices'", last_dim, kLessEqual,
                                         "the dimension of 'input_x'", input_x_size, prim_name);
  (void)CheckAndConvertUtils::CheckValue("len(updates.shape)'", updates_size, kEqual,
                                         "len(indices.shape) - 1 + len(input_x.shape) - indices.shape[-1]",
                                         indices_size - 1 + input_x_size - last_dim, prim_name);

  for (size_t i = 0; i < LongToSize(indices_size - 1); ++i) {
    (void)CheckAndConvertUtils::CheckValue<int64_t>(std::to_string(i) + "th dimension of indices", indices_shape[i],
                                                    kEqual, std::to_string(i) + "th dimension of updates",
                                                    updates_shape[i], prim_name);
  }
  for (int64_t i = indices_size - 1; i < updates_size; ++i) {
    (void)CheckAndConvertUtils::CheckValue<int64_t>(
      std::to_string(i) + "th dimension of updates", updates_shape[LongToSize(i)], kEqual,
      std::to_string(i - (indices_size - 1) + last_dim) + "th dimension of input_x.shape[indices.shape[-1]:]",
      input_x_shape[LongToSize(i - (indices_size - 1) + last_dim)], prim_name);
  }
  auto output_shape = input_x_shape_ptr->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr ScatterNdArithmeticInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_x_dtype = input_args[kInputIndex0]->BuildType();
  auto indices_dtype = input_args[kInputIndex1]->BuildType();
  auto updates_dtype = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indices_dtype, {kInt32, kInt64}, prim_name);
  std::map<std::string, TypePtr> type_dict = {{"input_x", input_x_dtype}, {"updates", updates_dtype}};
  // Only ScatterNdUpdate supports boolean type
  if (prim_name == prim::kPrimScatterNdUpdate->name()) {
    return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types_with_complex_and_bool, prim_name);
  }
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types, prim_name);
}
}  // namespace

void ScatterNdUpdate::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdUpdate::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdUpdate::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdAdd::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdAdd::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdAdd::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdSub::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdSub::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdSub::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdMul::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdMul::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdMul::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdDiv::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdDiv::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdDiv::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdMax::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdMax::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdMax::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void ScatterNdMin::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void ScatterNdMin::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ScatterNdMin::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ScatterNdUpdate, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdAdd, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdSub, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdMul, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdDiv, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdMax, BaseOperator);
MIND_API_OPERATOR_IMPL(ScatterNdMin, BaseOperator);
AbstractBasePtr ScatterNdArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = ScatterNdArithmeticInferType(primitive, input_args);
  auto infer_shape = ScatterNdArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGScatterNdArithmeticInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterNdArithmeticInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterNdArithmeticInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterNdArithmeticInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdUpdate, prim::kPrimScatterNdUpdate, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdAdd, prim::kPrimScatterNdAdd, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdSub, prim::kPrimScatterNdSub, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdMul, prim::kPrimScatterNdMul, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdDiv, prim::kPrimScatterNdDiv, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdMax, prim::kPrimScatterNdMax, AGScatterNdArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterNdMin, prim::kPrimScatterNdMin, AGScatterNdArithmeticInfer, false);
}  // namespace ops
}  // namespace mindspore
