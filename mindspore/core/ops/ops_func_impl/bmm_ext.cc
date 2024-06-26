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

#include "ops/ops_func_impl/bmm_ext.h"
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
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

BaseShapePtr BatchMatMulExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto constexpr kBatchMatmulExtInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, kBatchMatmulExtInputNum,
                                           primitive->name());
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a Tensor type, but got:" << input_args[0]->ToString();
  }
  if (y_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'y' must be a Tensor type, but got:" << input_args[1]->ToString();
  }

  auto x_shp = x_shape_map[kShape];
  auto y_shp = y_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(y_shp);
  if (!dynamic_shape) {
    if (x_shp.size() != kDim3) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'x' must be a 3D dimensional Tensor, but got " << x_shp.size()
                               << "D shape " << x_shp;
    }
    if (y_shp.size() != kDim3) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'y' must be a 3D dimensional Tensor, but got " << y_shp.size()
                               << "D shape " << y_shp;
    }
    int64_t x_col = x_shp[2];
    int64_t y_row = y_shp[1];
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                               << x_shp << ", y shape " << y_shp;
    }
    if (x_shp[0] != y_shp[0]) {
      MS_EXCEPTION(ValueError)
        << "For " << primitive->name()
        << ", one of the input's batch dim must be equal to another input's peer batch dim, but got " << x_shp[0]
        << " and " << y_shp[0] << ", with x shape " << x_shp << ", y shape " << y_shp;
    }
  }

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, y_shp, false, false);
  return std::make_shared<abstract::Shape>(ret_shape);
}
TypePtr BatchMatMulExtFuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->GetType());
  (void)types.emplace("w", input_args[1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[0]->GetType();
}

TypePtrList BatchMatMulExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  TypePtr ret_type = x_tensor->Dtype();
  const auto x_type = x_tensor->Dtype();
  const auto y_type = y_tensor->Dtype();
  auto op_name = primitive->name();
  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  return {ret_type};
}

ShapeArray BatchMatMulExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &x_shp = x_tensor->shape();
  const auto &y_shp = y_tensor->shape();

  if (x_shp.size() != kDim3) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the input 'x' must be a 3D dimensional Tensor, but got " << x_shp.size()
                             << "D shape " << x_shp;
  }
  if (y_shp.size() != kDim3) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the input 'y' must be a 3D dimensional Tensor, but got " << y_shp.size()
                             << "D shape " << y_shp;
  }
  int64_t x_col = x_shp[2];
  int64_t y_row = y_shp[1];
  if (x_col != y_row) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                             << x_shp << ", y shape " << y_shp;
  }
  if (x_shp[0] != y_shp[0]) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name()
      << ", one of the input's batch dim must be equal to another input's peer batch dim, but got " << x_shp[0]
      << " and " << y_shp[0] << ", with x shape " << x_shp << ", y shape " << y_shp;
  }

  ShapeVector ret_shape = {x_shp[0], x_shp[1], y_shp[2]};
  return {ret_shape};
}
REGISTER_SIMPLE_INFER(kNameBatchMatMulExt, BatchMatMulExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
