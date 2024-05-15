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

#include "ops/ops_func_impl/dot.h"
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {

BaseShapePtr DotFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shp = input_args[0]->GetShape()->GetShapeVector();
  auto y_shp = input_args[1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    ShapeVector ret_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(y_shp);
  if (!dynamic_shape) {
    if (x_shp.size() != kDim1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'input' must be a 1D dimensional Tensor, but got " << x_shp.size()
                               << "D shape " << x_shp;
    }
    if (y_shp.size() != kDim1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'other' must be a 1D dimensional Tensor, but got " << y_shp.size()
                               << "D shape " << y_shp;
    }
    int64_t x_col = x_shp[0];
    int64_t y_row = y_shp[0];
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError)
        << "For " << primitive->name()
        << ", the elements of the input 'input' should be same as the elements of the input 'other', with input shape "
        << x_shp << ", other shape " << y_shp;
    }
  }

  ShapeVector ret_shape;
  return std::make_shared<abstract::Shape>(ret_shape);
}
TypePtr DotFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[0]->GetType());
  (void)types.emplace("other", input_args[1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
