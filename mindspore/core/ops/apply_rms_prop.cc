/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/apply_rms_prop.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ApplyRMSProp, BaseOperator);
class ApplyRMSPropInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
    const int64_t kInputNum = 5;
    const int64_t kInputNumNormal = 8;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, op_name);
    auto var_shape = input_args[0]->BuildShape();
    auto ms_shape = input_args[1]->BuildShape();
    auto mom_shape = input_args[2]->BuildShape();
    auto grad_shape = input_args[4]->BuildShape();
    auto var_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto ms_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto mom_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
    auto grad_shape_map =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
    if (IsDynamicRank(var_shape_map) || IsDynamicRank(ms_shape_map) || IsDynamicRank(mom_shape_map) ||
        IsDynamicRank(grad_shape_map)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    if (var_shape->IsDynamic() || ms_shape->IsDynamic() || mom_shape->IsDynamic() || grad_shape->IsDynamic()) {
      return var_shape->cast<abstract::ShapePtr>();
    }
    // var and ms must have the same shape when is not dynamic
    if (*var_shape != *ms_shape) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', 'mean_square' must have the same shape as 'var'. But got 'mean_square' shape: "
                               << ms_shape->ToString() << ", 'var' shape: " << var_shape->ToString() << ".";
    }
    // var and mom must have the same shape when is not dynamic
    if (*var_shape != *mom_shape) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', 'moment' must have the same shape as 'var'. But got 'moment' shape: "
                               << mom_shape->ToString() << ", 'var' shape: " << var_shape->ToString() << ".";
    }
    // var and grad must have the same shape when is not dynamic
    if (*var_shape != *grad_shape) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', 'grad' must have the same shape as 'var'. But got 'grad' shape: "
                               << grad_shape->ToString() << ", 'var' shape: " << var_shape->ToString() << ".";
    }
    if (input_args.size() >= kInputNumNormal) {
      auto shape_element = var_shape->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape_element);
      return shape_element;
    } else {
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{var_shape, ms_shape, mom_shape});
    }
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t kInputNum = 5;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
    auto var_dtype = input_args[0]->BuildType();
    auto mean_square_dtype = input_args[1]->BuildType();
    auto moment_dtype = input_args[2]->BuildType();
    auto grad_dtype = input_args[4]->BuildType();
    auto learning_rate_dtype = input_args[3]->BuildType();
    auto decay_dtype = input_args[5]->BuildType();
    auto momentum_dtype = input_args[6]->BuildType();
    auto epsilon_dtype = input_args[7]->BuildType();
    std::map<std::string, TypePtr> types;
    (void)types.emplace("var dtype", var_dtype);
    (void)types.emplace("mean square dtype", mean_square_dtype);
    (void)types.emplace("moment dtype", moment_dtype);
    (void)types.emplace("grad dtype", grad_dtype);
    const std::set<TypePtr> number_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,   kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex64};
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, number_type, primitive->name());
    std::map<std::string, TypePtr> types_decay;
    (void)types_decay.emplace("decay dtype", decay_dtype);
    (void)types_decay.emplace("momentum dtype", momentum_dtype);
    (void)types_decay.emplace("epsilon dtype", epsilon_dtype);
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types_decay, valid_types, primitive->name());
    std::map<std::string, TypePtr> types_lr;
    (void)types_lr.emplace("learning rate dtype", learning_rate_dtype);
    (void)types_lr.emplace("decay dtype", decay_dtype);
    (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types_lr, valid_types, primitive->name(), true);
    return var_dtype;
  }
};
float ApplyRMSProp::get_attr(const char *attr) const {
  auto attr_ptr = GetAttr(attr);
  return GetValue<float>(attr_ptr);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyRMSProp, prim::kPrimApplyRMSProp, ApplyRMSPropInfer, false);
}  // namespace ops
}  // namespace mindspore
