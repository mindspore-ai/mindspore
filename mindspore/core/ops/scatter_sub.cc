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
#include "ops/scatter_sub.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScatterSubInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  BaseShapePtr input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  BaseShapePtr indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  BaseShapePtr updates_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(updates_shape_ptr);

  if (input_x_shape_ptr->IsDynamic() || indices_shape_ptr->IsDynamic() || updates_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  std::vector<int64_t> input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  std::vector<int64_t> indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  std::vector<int64_t> updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];
  ShapeVector check_update_shape(indices_shape);
  for (int64_t i = 1; i < SizeToLong(input_x_shape.size()); ++i) {
    check_update_shape.push_back(input_x_shape[LongToSize(i)]);
  }
  if (updates_shape != check_update_shape) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', "
                             << "updates_shape = indices_shape + input_x_shape[1:], but got input_x_shape: "
                             << input_x_shape_ptr->ToString() << ", indices_shape: " << indices_shape_ptr->ToString()
                             << ", updates_shape: " << updates_shape_ptr->ToString() << ".";
  }

  auto output_shape = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr ScatterSubInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_type_ptr = input_args[kInputIndex0]->BuildType();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  auto updates_type_ptr = input_args[kInputIndex2]->BuildType();
  auto prim_name = primitive->name();
  std::set<TypePtr> type_set = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indiecs_type_ptr, type_set, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_x_type_ptr, common_valid_types_with_complex,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("updates", updates_type_ptr, common_valid_types_with_complex,
                                                   prim_name);

  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input_x", input_x_type_ptr);
  (void)type_dict.emplace("updates", updates_type_ptr);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(ScatterSub, BaseOperator);
AbstractBasePtr ScatterSubInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = ScatterSubInferType(primitive, input_args);
  auto infer_shape = ScatterSubInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGScatterSubInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterSubInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterSubInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScatterSubInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScatterSub, prim::kPrimScatterSub, AGScatterSubInfer, false);
}  // namespace ops
}  // namespace mindspore
