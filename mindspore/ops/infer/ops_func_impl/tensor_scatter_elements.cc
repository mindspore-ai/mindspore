/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/tensor_scatter_elements.h"
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr TensorScatterElementsFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto indices_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto updates_shape_ptr = input_args[kInputIndex2]->GetShape();
  const auto &input_x_shape = input_x_shape_ptr->GetShapeVector();
  const auto &indices_shape = indices_shape_ptr->GetShapeVector();
  const auto &updates_shape = updates_shape_ptr->GetShapeVector();
  if (IsDynamicRank(input_x_shape)) {
    size_t rank;
    if (!IsDynamicRank(indices_shape)) {
      rank = indices_shape.size();
    } else if (!IsDynamicRank(updates_shape)) {
      rank = updates_shape.size();
    } else {
      return input_x_shape_ptr->Clone();
    }
    ShapeVector output_shape(rank, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(std::move(output_shape));
  }

  if (input_x_shape.size() < 1 || indices_shape.size() < 1 || updates_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", 'input_x_shape', 'indices_shape' and "
                             << "'updates_shape' dims must be greater than 1. but got input_x_shape:" << input_x_shape
                             << ", indices_shape:" << indices_shape << ", updates_shape: " << updates_shape << ".";
  }

  if (IsDynamicRank(indices_shape) || IsDynamicRank(updates_shape)) {
    return input_x_shape_ptr->Clone();
  }

  if (input_x_shape.size() != indices_shape.size() || input_x_shape.size() != updates_shape.size() ||
      indices_shape.size() != updates_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of 'input_x', 'indice' and 'update' should be same, but got "
                             << "input_x shape: " << input_x_shape << "; "
                             << "indice shape: " << indices_shape << "; "
                             << "update shape: " << updates_shape << ".";
  }

  return input_x_shape_ptr->Clone();
}

TypePtr TensorScatterElementsFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->GetType();
  std::set<TypePtr> type_set = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input_x", input_args[kInputIndex0]->GetType());
  (void)type_dict.emplace("updates", input_args[kInputIndex2]->GetType());
  std::set<TypePtr> check_list(common_valid_types);
  (void)check_list.insert(kBool);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, check_list, prim_name);
}
}  // namespace ops
}  // namespace mindspore
