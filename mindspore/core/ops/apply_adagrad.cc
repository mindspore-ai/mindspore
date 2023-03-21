/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "ops/apply_adagrad.h"

#include <set>
#include <utility>
#include <map>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
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
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ApplyAdagrad, BaseOperator);

class ApplyAdagradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kInputNum = 4;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
    auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
    auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
    auto var_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto accum_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto grad_shape_ptr = input_args[kInputIndex3]->BuildShape();
    if (IsDynamicRank(var_shape) || IsDynamicRank(accum_shape)) {
      auto unknow_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr});
    }
    if (IsDynamicRank(grad_shape)) {
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
    }
    if (grad_shape_ptr->IsDynamic() || accum_shape_ptr->IsDynamic() || var_shape_ptr->IsDynamic()) {
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
    }
    // lr must be scalar or size equal with 1
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kLessEqual, 1, prim_name);
    if (lr_shape.size() == 1) {
      (void)CheckAndConvertUtils::CheckInteger("lr_shape's first rank must be 1", lr_shape[0], kEqual, 1, prim_name);
    }
    // var, accum and grad must have the same shape
    std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
    (void)same_shape_args_map.insert(std::make_pair("accum", accum_shape_ptr));
    (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape_ptr));
    for (auto &elem : same_shape_args_map) {
      if (*elem.second != *var_shape_ptr) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                                 << "' and 'var' must have the same shape. But got '" << elem.first
                                 << "' shape: " << elem.second->ToString()
                                 << ", 'var' shape: " << var_shape_ptr->ToString() << ".";
      }
    }
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kInputNum = 4;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
    auto var_type = input_args[kInputIndex0]->BuildType();
    auto accum_type = input_args[kInputIndex1]->BuildType();
    auto lr_type = input_args[kInputIndex2]->BuildType();
    auto grad_type = input_args[kInputIndex3]->BuildType();
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    // var, accum and grad must have the same type
    std::map<std::string, TypePtr> args;
    (void)args.insert(std::make_pair("var", var_type));
    (void)args.insert(std::make_pair("accum", accum_type));
    (void)args.insert(std::make_pair("grad", grad_type));
    (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
    // lr type must be valid
    std::map<std::string, TypePtr> args_lr;
    (void)args_lr.insert(std::make_pair("lr", lr_type));
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
    return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
  }
};
bool ApplyAdagrad::get_update_slots() const {
  auto value_ptr = this->GetAttr(kUpdateSlots);
  return GetValue<bool>(value_ptr);
}
void ApplyAdagrad::set_update_slots(const bool update_slots) {
  (void)this->AddAttr(kUpdateSlots, api::MakeValue(update_slots));
}

abstract::AbstractBasePtr ApplyAdagradInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  ApplyAdagradInfer apply_ada_grad;
  auto type = apply_ada_grad.InferType(primitive, input_args);
  auto shape = apply_ada_grad.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyAdagrad, prim::kPrimApplyAdagrad, ApplyAdagradInfer, false);
}  // namespace ops
}  // namespace mindspore
