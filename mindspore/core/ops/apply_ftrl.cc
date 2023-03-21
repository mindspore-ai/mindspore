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

#include "ops/apply_ftrl.h"

#include <set>
#include <utility>
#include <map>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ApplyFtrl, BaseOperator);
class ApplyFtrlInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kInputNum = 8;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto var_shape = input_args[kInputIndex0]->BuildShape();
    auto accum_shape = input_args[kInputIndex1]->BuildShape();
    auto linear_shape = input_args[kInputIndex2]->BuildShape();
    auto grad_shape = input_args[kInputIndex3]->BuildShape();
    auto lr_shape = input_args[kInputIndex4]->BuildShape();
    auto l1_shape = input_args[kInputIndex5]->BuildShape();
    auto l2_shape = input_args[kInputIndex6]->BuildShape();
    auto lr_power_shape = input_args[kInputIndex7]->BuildShape();
    auto var_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto accum_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto linear_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
    auto grad_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
    auto lr_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
    auto l1_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
    auto l2_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
    auto lr_power_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
    int64_t batch_rank = 0;
    if (IsDynamicRank(var_shape_map) || IsDynamicRank(accum_shape_map) || IsDynamicRank(grad_shape_map) ||
        IsDynamicRank(linear_shape_map)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    if (var_shape->IsDynamic() || accum_shape->IsDynamic() || linear_shape->IsDynamic() || grad_shape->IsDynamic() ||
        lr_shape->IsDynamic() || l1_shape->IsDynamic() || l2_shape->IsDynamic() || lr_power_shape->IsDynamic()) {
      return var_shape->cast<abstract::ShapePtr>();
    }
    if (primitive->HasAttr(kBatchRank)) {
      auto value_ptr = primitive->GetAttr(kBatchRank);
      batch_rank = GetValue<int64_t>(value_ptr);
    }
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape_map.size()), kGreaterEqual,
                                             batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToLong(l1_shape_map.size()), kGreaterEqual,
                                             batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToLong(l2_shape_map.size()), kGreaterEqual,
                                             batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("lr_power_shape size", SizeToLong(lr_power_shape_map.size()),
                                             kGreaterEqual, batch_rank, prim_name);
    std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
    (void)same_shape_args_map.insert(std::make_pair("accum", accum_shape));
    (void)same_shape_args_map.insert(std::make_pair("linear", linear_shape));
    (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape));
    for (auto &elem : same_shape_args_map) {
      if (*elem.second != *var_shape) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                                 << "' must have the same shape as 'var'. But got '" << elem.first
                                 << "' shape:  " << elem.second->ToString()
                                 << ", 'var' shape: " << var_shape->ToString() << ".";
      }
    }
    auto shape_ptr = var_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr;
  }
  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t kInputNum = 8;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto var_type = input_args[kInputIndex0]->BuildType();
    auto accum_type = input_args[kInputIndex1]->BuildType();
    auto linear_type = input_args[kInputIndex2]->BuildType();
    auto grad_type = input_args[kInputIndex3]->BuildType();
    const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    std::map<std::string, TypePtr> args;
    (void)args.insert(std::make_pair("var_type", var_type));
    (void)args.insert(std::make_pair("accum_type", accum_type));
    (void)args.insert(std::make_pair("linear_type", linear_type));
    (void)args.insert(std::make_pair("grad_type", grad_type));
    // var accum linear grad must have same dtypes
    (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

    auto lr_type = input_args[kInputIndex4]->BuildType();
    auto l1_type = input_args[kInputIndex5]->BuildType();
    auto l2_type = input_args[kInputIndex6]->BuildType();
    auto lr_power_type = input_args[kInputIndex7]->BuildType();
    std::map<std::string, TypePtr> args_lr;
    std::map<std::string, TypePtr> args_l1;
    std::map<std::string, TypePtr> args_l2;
    std::map<std::string, TypePtr> args_lr_power;
    (void)args_lr.insert(std::make_pair("lr_type", lr_type));
    (void)args_l1.insert(std::make_pair("l1_type", l1_type));
    (void)args_l2.insert(std::make_pair("l2_type", l2_type));
    (void)args_lr_power.insert(std::make_pair("lr_power_type", lr_power_type));

    // lr, l1, l2, lr_power type must be float or scalar tensor with float
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr_power, valid_types, prim_name);

    return var_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyFtrl, prim::kPrimApplyFtrl, ApplyFtrlInfer, false);
}  // namespace ops
}  // namespace mindspore
