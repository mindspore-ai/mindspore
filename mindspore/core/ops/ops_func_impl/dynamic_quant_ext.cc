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
#include "ops/ops_func_impl/dynamic_quant_ext.h"
#include <vector>
#include <memory>
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"

namespace mindspore::ops {
BaseShapePtr DynamicQuantExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  auto input_shape_vector = input_shape->GetShapeVector();
  std::vector<BaseShapePtr> shapes_list;
  shapes_list.push_back(input_shape->Clone());
  if (IsDynamicRank(input_shape_vector)) {
    ShapeVector scale_shape = {abstract::Shape::kShapeRankAny};
    shapes_list.push_back(std::make_shared<abstract::Shape>(scale_shape));
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  if (input_shape_vector.size() < 1) {
    MS_EXCEPTION(ValueError) << "Input shape size should be at least 1, but got " << input_shape_vector.size();
  }

  ShapeVector scale_shape(input_shape_vector.begin(), input_shape_vector.end() - 1);
  shapes_list.push_back(std::make_shared<abstract::Shape>(scale_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr DynamicQuantExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  std::vector<TypePtr> types_list = {kInt8, kFloat32};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace mindspore::ops
