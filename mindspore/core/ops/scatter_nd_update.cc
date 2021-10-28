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

#include "ops/scatter_nd_update.h"

#include <map>
#include <set>
#include <string>

#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScatterNdUpdateInferShape(const PrimitivePtr &primitive,
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
  if (indices_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("indices_shape", indices_shape[0], kNotEqual, -1);
  }
  auto last_dim = indices_shape.back();
  indices_shape.pop_back();
  indices_shape.insert(indices_shape.end(), input_x_shape.begin() + last_dim, input_x_shape.end());
  (void)CheckAndConvertUtils::CheckInteger("length of indices_shape[:-1] + x_shape[indices_shape[-1]:]",
                                           updates_shape.size(), kEqual, indices_shape.size(), prim_name);
  for (size_t i = 0; i < updates_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger("elements of indices_shape[:-1] + x_shape[indices_shape[-1]:]",
                                             updates_shape[i], kEqual, indices_shape[i], prim_name);
  }
  return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
}

TypePtr ScatterNdUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> type_set = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  type_dict.emplace("input_x", input_args[kInputIndex0]->BuildType());
  type_dict.emplace("updates", input_args[kInputIndex2]->BuildType());
  std::set<TypePtr> check_list(common_valid_types);
  check_list.insert(kBool);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, check_list, prim_name);
}
}  // namespace

AbstractBasePtr ScatterNdUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = ScatterNdUpdateInferType(primitive, input_args);
  auto infer_shape = ScatterNdUpdateInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterNdUpdate, prim::kPrimScatterNdUpdate, ScatterNdUpdateInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
