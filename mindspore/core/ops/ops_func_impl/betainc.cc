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
#include "ops/ops_func_impl/betainc.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BetaincFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  ShapeArray all_input_shapes{};
  auto a_shape_base = input_args[kInputIndex0]->GetShape();
  auto b_shape_base = input_args[kInputIndex1]->GetShape();
  auto x_shape_base = input_args[kInputIndex2]->GetShape();
  const auto &a_shape = a_shape_base->GetShapeVector();
  all_input_shapes.emplace_back(a_shape);
  const auto &b_shape = b_shape_base->GetShapeVector();
  all_input_shapes.emplace_back(b_shape);
  const auto &x_shape = x_shape_base->GetShapeVector();
  all_input_shapes.emplace_back(x_shape);
  if (IsDynamicRank(a_shape) && IsDynamicRank(b_shape) && IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  if (!(IsDynamic(a_shape) || IsDynamic(b_shape))) {
    if (a_shape != b_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << ", shape of b " << b_shape_base->ToString()
                               << " are not consistent with the shape a " << a_shape_base->ToString() << " .";
    }
  }
  if (!(IsDynamic(a_shape) || IsDynamic(x_shape))) {
    if (a_shape != x_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << ", shape of x " << x_shape_base->ToString()
                               << " are not consistent with the shape a " << a_shape_base->ToString() << " .";
    }
  }
  if (!(IsDynamic(b_shape) || IsDynamic(x_shape))) {
    if (b_shape != x_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << ", shape of x " << x_shape_base->ToString()
                               << " are not consistent with the shape b " << b_shape_base->ToString() << " .";
    }
  }
  auto out_shape = InferOutShapeSameAsInShape(all_input_shapes);
  return std::make_shared<abstract::Shape>(std::move(out_shape));
}

TypePtr BetaincFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto a_type = input_args[kInputIndex0]->GetType();
  auto b_type = input_args[kInputIndex1]->GetType();
  auto x_type = input_args[kInputIndex2]->GetType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> args_type;
  (void)args_type.emplace("a", a_type);
  (void)args_type.emplace("b", b_type);
  (void)args_type.emplace("x", x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_type, valid_types, primitive->name());
  return a_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
