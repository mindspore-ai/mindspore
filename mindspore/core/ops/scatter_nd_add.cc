/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/scatter_nd_add.h"

#include <map>
#include <set>
#include <string>

#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScatterNdAddInferShape(const PrimitivePtr &primitive,
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

  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (indices_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("indices_shape", indices_shape[0], kNotEqual, -1);
  }
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]->BuildShape());
  auto last_dim = indices_shape.back();
  indices_shape.pop_back();
  indices_shape.insert(indices_shape.end(), input_x_shape.begin() + last_dim, input_x_shape.end());
  (void)CheckAndConvertUtils::CheckInteger("length of updates_shape and indices_shape + x_shape[1:]",
                                           updates_shape.size(), kEqual, indices_shape.size(), prim_name);
  for (size_t i = 0; i < updates_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger("elements of updates_shape and indices_shape + x_shape[1:]",
                                             updates_shape[i], kEqual, indices_shape[i], prim_name);
  }
  auto output_shape = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr ScatterNdAddInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> type_set = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  type_dict.emplace("input_x", input_args[kInputIndex0]->BuildType());
  type_dict.emplace("updates", input_args[kInputIndex2]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types, prim_name);
}
}  // namespace

AbstractBasePtr ScatterNdAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = ScatterNdAddInferType(primitive, input_args);
  auto infer_shape = ScatterNdAddInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterNdAdd, prim::kPrimScatterNdAdd, ScatterNdAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
