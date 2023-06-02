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

#include "ops/scatter_arithmetic.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "ops/scatter_add.h"
#include "ops/scatter_div.h"
#include "ops/scatter_max.h"
#include "ops/scatter_min.h"
#include "ops/scatter_mul.h"
#include "ops/scatter_update.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScatterArithmeticInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  auto updates_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(updates_shape_ptr);
  if (input_x_shape_ptr->IsDynamic() || indices_shape_ptr->IsDynamic() || updates_shape_ptr->IsDynamic()) {
    return input_x_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];
  std::vector<int64_t> check_update_shape(indices_shape);
  for (size_t i = 1; i < input_x_shape.size(); ++i) {
    check_update_shape.push_back(input_x_shape[i]);
  }
  if (updates_shape != check_update_shape) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", "
                             << "updates_shape = indices_shape + input_x_shape[1:], but got input_x_shape: "
                             << input_x_shape_ptr->ToString() << ", indices_shape: " << indices_shape_ptr->ToString()
                             << ", updates_shape: " << updates_shape_ptr->ToString() << ".";
  }

  return input_x_shape_ptr->cast<abstract::ShapePtr>();
}

TypePtr ScatterArithmeticInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_type_ptr = input_args[kInputIndex0]->BuildType();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  auto updates_type_ptr = input_args[kInputIndex2]->BuildType();
  auto prim_name = primitive->name();
  const std::set<TypePtr> indices_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, indices_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x type", input_x_type_ptr, common_valid_types_with_complex,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("updates type", updates_type_ptr, common_valid_types_with_complex,
                                                   prim_name);

  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input_x", input_x_type_ptr);
  (void)type_dict.emplace("updates", updates_type_ptr);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types_with_complex, prim_name);
}
}  // namespace

AbstractBasePtr ScatterArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = ScatterArithmeticInferType(primitive, input_args);
  auto infer_shape = ScatterArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstractTensor(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ScatterAdd, BaseOperator);

// AG means auto generated
class MIND_API AGScatterArithmeticInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterArithmeticInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterArithmeticInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterArithmeticInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterAdd, prim::kPrimScatterAdd, AGScatterArithmeticInfer, false);
MIND_API_OPERATOR_IMPL(ScatterUpdate, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterUpdate, prim::kPrimScatterUpdate, AGScatterArithmeticInfer, false);

MIND_API_OPERATOR_IMPL(ScatterMin, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterMin, prim::kPrimScatterMin, AGScatterArithmeticInfer, false);

MIND_API_OPERATOR_IMPL(ScatterMax, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterMax, prim::kPrimScatterMax, AGScatterArithmeticInfer, false);

MIND_API_OPERATOR_IMPL(ScatterDiv, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterDiv, prim::kPrimScatterDiv, AGScatterArithmeticInfer, false);

MIND_API_OPERATOR_IMPL(ScatterMul, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterMul, prim::kPrimScatterMul, AGScatterArithmeticInfer, false);
}  // namespace ops
}  // namespace mindspore
