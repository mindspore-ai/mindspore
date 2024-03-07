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

#include "ops/ops_func_impl/quant_batch_matmul.h"
#include <algorithm>
#include <map>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kQbmmMatSize = 2;
constexpr size_t kQbmmInputX1 = 0;
constexpr size_t kQbmmInputX2 = 1;
constexpr size_t kQbmmInputScale = 2;
constexpr size_t kQbmmInputOffset = 3;
constexpr size_t kQbmmInputBias = 4;
constexpr size_t kQbmmInputTransposeX1 = 5;
constexpr size_t kQbmmInputTransposeX2 = 6;
constexpr size_t kQbmmInputDtype = 7;
}  // namespace
BaseShapePtr QuantBatchMatmulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x1_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kQbmmInputX1]->GetShape());
  auto x2_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kQbmmInputX2]->GetShape());
  if (x1_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X1' must be a Tensor type, but got:" << input_args[kQbmmInputX1]->ToString();
  }
  if (x2_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X2' must be a Tensor type, but got:" << input_args[kQbmmInputX2]->ToString();
  }

  ValuePtr transpose_x1_ptr = input_args[kQbmmInputTransposeX1]->GetValue();
  ValuePtr transpose_x2_ptr = input_args[kQbmmInputTransposeX2]->GetValue();
  bool transpose_x1 = GetValue<bool>(transpose_x1_ptr);
  bool transpose_x2 = GetValue<bool>(transpose_x2_ptr);

  auto x1_shp = x1_shape_map[kShape];
  auto x2_shp = x2_shape_map[kShape];
  if (IsDynamicRank(x1_shp) || IsDynamicRank(x2_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  bool dynamic_shape = IsDynamic(x1_shp) || IsDynamic(x2_shp);
  if (!dynamic_shape) {
    CheckBatchMatmulInputSize(prim_name, "x", x1_shp);
    CheckBatchMatmulInputSize(prim_name, "y", x2_shp);
    CheckBatchMatmulInputWhetherCanBeMul(prim_name, x1_shp, x2_shp, transpose_x1, transpose_x2);
    CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x1_shp, x2_shp);
  }

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x1_shp, x2_shp, transpose_x1, transpose_x2, kQbmmMatSize);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr QuantBatchMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[kQbmmInputX1]);
  TypePtr x1_type = input_args[kQbmmInputX1]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kQbmmInputX2]);
  TypePtr x2_type = input_args[kQbmmInputX2]->GetType();
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt8}, primitive->name());

  types.clear();
  MS_EXCEPTION_IF_NULL(input_args[kQbmmInputScale]);
  TypePtr scale_type = input_args[kQbmmInputScale]->GetType();
  (void)types.emplace("scale", scale_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32, kUInt64, kInt64, kBFloat16}, primitive->name());

  MS_EXCEPTION_IF_NULL(input_args[kQbmmInputOffset]);
  if (!input_args[kQbmmInputOffset]->GetType()->isa<TypeNone>()) {
    types.clear();
    (void)types.emplace("offset", input_args[kQbmmInputOffset]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, primitive->name());
  }

  MS_EXCEPTION_IF_NULL(input_args[kQbmmInputBias]);
  if (!input_args[kQbmmInputBias]->GetType()->isa<TypeNone>()) {
    types.clear();
    (void)types.emplace("bias", input_args[kQbmmInputBias]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32}, primitive->name());
  }

  ValuePtr dtype_ptr = input_args[kQbmmInputDtype]->GetValue();
  auto dtype = GetValue<int64_t>(dtype_ptr);
  if (dtype == TypeId::kNumberTypeInt8) {
    return kInt8;
  } else {
    return kFloat16;
  }
}

void QuantBatchMatmulFuncImpl::CheckBatchMatmulInputWhetherCanBeMul(const std::string &name,
                                                                    const ShapeVector &x1_shape,
                                                                    const ShapeVector &x2_shape, bool transpose_x1,
                                                                    bool transpose_x2) const {
  ShapeVector x1_mat_shape(x1_shape.end() - SizeToLong(kQbmmMatSize), x1_shape.end());
  ShapeVector x2_mat_shape(x2_shape.end() - SizeToLong(kQbmmMatSize), x2_shape.end());
  int64_t x1_col = x1_mat_shape[static_cast<size_t>(!transpose_x1)];
  int64_t x2_row = x2_mat_shape[static_cast<size_t>(transpose_x2)];
  if (std::find(x1_shape.begin(), x1_shape.end(), -1) == x1_shape.end() &&
      std::find(x2_shape.begin(), x2_shape.end(), -1) == x2_shape.end()) {
    if (x1_col != x2_row) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                               << x1_shape << "(transpose_x1=" << transpose_x1 << "), y shape " << x2_shape
                               << "(transpose_x2=" << transpose_x2 << ")";
    }
  }
}

void QuantBatchMatmulFuncImpl::BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp,
                                                    bool transpose_x1, bool transpose_x2, size_t offset) const {
  if (xshp.empty() || yshp.empty()) {
    return;
  }
  ShapeVector long_input = xshp.size() > yshp.size() ? xshp : yshp;
  ShapeVector short_input = xshp.size() > yshp.size() ? yshp : xshp;
  size_t size_diff = long_input.size() - short_input.size();
  for (size_t i = 0; i < long_input.size() - offset; i++) {
    if (long_input[i] < 0) {
      output->push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      output->push_back(long_input[i] > short_input[i - size_diff] ? long_input[i] : short_input[i - size_diff]);
    } else {
      output->push_back(long_input[i]);
    }
  }
  size_t x1_offset = xshp.size() - offset;
  size_t x2_offset = yshp.size() - offset;
  output->push_back(xshp[x1_offset + (transpose_x1 ? 1 : 0)]);
  output->push_back(yshp[x2_offset + (transpose_x2 ? 0 : 1)]);
  return;
}

void QuantBatchMatmulFuncImpl::CheckBatchMatmulInputSize(const std::string &op_name, const std::string &input_name,
                                                         const ShapeVector &shape) const {
  constexpr size_t dim_limit = 2;
  if (shape.size() < dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D or higher dimensional Tensor, but got " << shape.size() << "D shape "
                             << shape;
  }
}

void QuantBatchMatmulFuncImpl::CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name,
                                                                          const ShapeVector &x1_shape,
                                                                          const ShapeVector &x2_shape) const {
  ShapeVector x1_batch(x1_shape.begin(), x1_shape.end() - SizeToLong(kQbmmMatSize));
  ShapeVector x2_batch(x2_shape.begin(), x2_shape.end() - SizeToLong(kQbmmMatSize));
  if (x1_batch == x2_batch) {
    return;
  }

  size_t min_size = std::min(x1_batch.size(), x2_batch.size());
  for (int64_t i = 0; i < SizeToLong(min_size); ++i) {
    auto x = *(x1_batch.rbegin() + i);
    auto y = *(x2_batch.rbegin() + i);
    if (x != 1 && y != 1 && x != y) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", one of the input's batch dim must be equal to another input's peer batch dim, or "
                                  "be equal to 1, or be empty, but got "
                               << x << " and " << y << ", with x shape " << x1_shape << ", y shape " << x2_shape;
    }
  }
}

}  // namespace ops
}  // namespace mindspore
