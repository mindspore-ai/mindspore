/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/eig.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kDefaultRank = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

void EigCheckShapeValid(const ShapeVector &input_shape) {
  if (IsDynamicRank(input_shape)) {
    return;
  }

  if (input_shape[input_shape.size() - kRowIndex] == abstract::Shape::kShapeDimAny ||
      input_shape[input_shape.size() - kColIndex] == abstract::Shape::kShapeDimAny) {
    return;
  }

  if (input_shape[input_shape.size() - kRowIndex] != input_shape[input_shape.size() - kColIndex]) {
    MS_EXCEPTION(ValueError) << "For Eig, x should be square(squares)"
                             << ", but got " << input_shape[input_shape.size() - kRowIndex] << " × "
                             << input_shape[input_shape.size() - kColIndex] << " matrix(matrices).";
  }
}

BaseShapePtr EigFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  std::vector<BaseShapePtr> shapes_list(kDefaultRank);
  EigCheckShapeValid(input_shape);

  /* infer eigen_value shape  */
  if (IsDynamicRank(input_shape)) {
    shapes_list.at(kIndex0) = std::make_shared<abstract::Shape>(input_shape);
  } else {
    ShapeVector val_shape_list;
    val_shape_list.assign(input_shape.begin(), input_shape.end());
    int col_value = input_shape.back();
    val_shape_list.pop_back();
    if (val_shape_list.back() == abstract::Shape::kShapeDimAny) {
      val_shape_list.back() = col_value;
    }
    shapes_list.at(kIndex0) = std::make_shared<abstract::Shape>(val_shape_list);
  }

  /* infer eigen_vectors shape  */
  auto compute_v_scalar = GetScalarValue<bool>(input_args[kInputIndex1]->GetValue());
  if (!compute_v_scalar.has_value()) {
    shapes_list.at(kIndex1) = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  } else {
    auto compute_v_value = compute_v_scalar.value();
    if (compute_v_value) {
      shapes_list.at(kIndex1) = std::make_shared<abstract::Shape>(input_shape);
    } else {
      shapes_list.at(kIndex1) = std::make_shared<abstract::Shape>(std::vector<int64_t>{});
    }
  }

  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr EigFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  std::vector<TypePtr> types_list;
  if (*(x_type->cast<TensorTypePtr>()->element()) == *(kFloat32)) {
    types_list = {std::make_shared<TensorType>(kComplex64), std::make_shared<TensorType>(kComplex64)};
  } else if (*(x_type->cast<TensorTypePtr>()->element()) == *(kFloat64)) {
    types_list = {std::make_shared<TensorType>(kComplex128), std::make_shared<TensorType>(kComplex128)};
  } else {
    types_list = {x_type, x_type};
  }
  return std::make_shared<Tuple>(types_list);
}
}  // namespace ops
}  // namespace mindspore
