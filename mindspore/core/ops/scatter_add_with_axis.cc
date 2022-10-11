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

#include "ops/scatter_add_with_axis.h"

#include <map>
#include <set>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScatterAddWithAxisInferShape(const PrimitivePtr &primitive,
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
  if (input_x_shape.size() < 1 || indices_shape.size() < 1 || updates_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", 'input_x_shape', 'indices_shape' and "
                             << "'updates_shape' dims must be greater than 1. but got input_x_shape:" << input_x_shape
                             << ", indices_shape:" << indices_shape << ", updates_shape: " << updates_shape << ".";
  }
  if (updates_shape != indices_shape) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", "
                             << "'updates_shape' must be as same as 'indices_shape' but got "
                                "indices_shape: "
                             << indices_shape << ", updates_shape: " << updates_shape << ".";
  }

  return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
}

TypePtr ScatterAddWithAxisInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> type_set = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input_x", input_args[kInputIndex0]->BuildType());
  (void)type_dict.emplace("updates", input_args[kInputIndex2]->BuildType());
  std::set<TypePtr> check_list(common_valid_types);
  (void)check_list.insert(kBool);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, check_list, prim_name);
}
}  // namespace

AbstractBasePtr ScatterAddWithAxisInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = ScatterAddWithAxisInferType(primitive, input_args);
  auto infer_shape = ScatterAddWithAxisInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ScatterAddWithAxis, BaseOperator);
void ScatterAddWithAxis::Init(const int64_t axis) { this->set_axis(axis); }
void ScatterAddWithAxis::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t ScatterAddWithAxis::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterAddWithAxis, prim::kPrimScatterAddWithAxis, ScatterAddWithAxisInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
