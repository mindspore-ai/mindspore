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

#include "ops/ops_func_impl/addmm.h"
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
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

BaseShapePtr AddmmFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shp = input_args[kIndex1]->GetShape()->GetShapeVector();
  auto y_shp = input_args[kIndex2]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    ShapeVector ret_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(y_shp);
  if (!dynamic_shape) {
    if (x_shp.size() != kDim2) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'mat1' must be a 2D dimensional Tensor, but got " << x_shp.size()
                               << "D shape " << x_shp;
    }
    if (y_shp.size() != kDim2) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'mat2' must be a 2D dimensional Tensor, but got " << y_shp.size()
                               << "D shape " << y_shp;
    }
    int64_t x_col = x_shp[kIndex1];
    int64_t y_row = y_shp[kIndex0];
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError)
        << "For " << primitive->name()
        << ", the elements of the input 'mat1' should be same as the elements of the input 'mat2', with input shape "
        << x_shp << ", other shape " << y_shp;
    }
  }

  ShapeVector ret_shape = {x_shp[kIndex0], y_shp[kIndex1]};
  return std::make_shared<abstract::Shape>(ret_shape);
}
TypePtr AddmmFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("mat1", input_args[kIndex1]->GetType());
  (void)types.emplace("mat2", input_args[kIndex2]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[0]->GetType();
}
TypePtrList AddmmFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_type = x_tensor->Dtype();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, valid_types, primitive->name());
  return {x_type};
}
ShapeArray AddmmFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &y_tensor = input_values[kIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);
  const auto &x_shape = x_tensor->shape();
  const auto &y_shape = y_tensor->shape();
  if (x_shape.size() != kDim2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the input 'mat1' must be a 2D dimensional Tensor, but got " << x_shape.size()
                             << "D shape " << x_shape;
  }
  if (y_shape.size() != kDim2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the input 'mat2' must be a 2D dimensional Tensor, but got " << y_shape.size()
                             << "D shape " << y_shape;
  }

  int64_t x_col = x_shape[kIndex1];
  int64_t y_row = y_shape[kIndex0];
  if (x_col != y_row) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name()
      << ", the elements of the input 'mat1' should be same as the elements of the input 'mat2', with input shape "
      << x_shape << ", other shape " << y_shape;
  }
  ShapeVector ret_shape = {x_shape[0], y_shape[1]};
  return {ret_shape};
}
REGISTER_SIMPLE_INFER(kNameAddmm, AddmmFuncImpl)
}  // namespace ops
}  // namespace mindspore
