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

#include "ops/ops_func_impl/weight_quant_batch_matmul.h"
#include <algorithm>
#include <map>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kMatSize = 2;
constexpr size_t kInputX = 0;
constexpr size_t kInputWeight = 1;
constexpr size_t kInputAntiquantScale = 2;
constexpr size_t kInputAntiquantOffset = 3;
constexpr size_t kInputQuantScale = 4;
constexpr size_t kInputQuantOffset = 5;
constexpr size_t kInputBias = 6;
constexpr size_t kInputTransposeX = 7;
constexpr size_t kInputTransposeWeight = 8;
constexpr size_t kInputGroupSize = 9;
constexpr int kInt4ShapeMul = 2;
}  // namespace

BaseShapePtr WeightQuantBatchMatmulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);

  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputX]->GetShape());
  auto weight_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputWeight]->GetShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X' must be a Tensor type, but got:" << input_args[kInputX]->ToString();
  }
  if (weight_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'Weight' must be a Tensor type, but got:" << input_args[kInputWeight]->ToString();
  }

  ValuePtr transpose_x_ptr = input_args[kInputTransposeX]->GetValue();
  ValuePtr transpose_weight_ptr = input_args[kInputTransposeWeight]->GetValue();
  auto transpose_x_any = GetScalarValue<bool>(transpose_x_ptr);
  if (!transpose_x_any.has_value()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'transpose_x' has no value:" << input_args[kInputTransposeX]->ToString();
  }
  bool transpose_x = transpose_x_any.value();
  auto transpose_weight_any = GetScalarValue<bool>(transpose_weight_ptr);
  if (!transpose_weight_any.has_value()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'transpose_weight' has no value:" << input_args[kInputTransposeWeight]->ToString();
  }
  bool transpose_weight = transpose_weight_any.value();

  auto x_shp = x_shape_map[kShape];
  auto weight_shp = weight_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(weight_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(weight_shp);
  auto input_type = input_args[kInputWeight]->GetType()->cast<TensorTypePtr>()->element();
  if (!dynamic_shape) {
    CheckBatchMatmulInputSize(prim_name, "x", x_shp);
    CheckBatchMatmulInputSize(prim_name, "weight", weight_shp);
    CheckBatchMatmulInputWhetherCanBeMul(prim_name, x_shp, weight_shp, transpose_x, transpose_weight,
                                         input_type->type_id());
    CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x_shp, weight_shp);
  }

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, weight_shp, transpose_x, transpose_weight, kMatSize);
  if (input_type->type_id() == TypeId::kNumberTypeInt4 && !transpose_weight) {
    ret_shape[ret_shape.size() - 1] *= kInt4ShapeMul;
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr WeightQuantBatchMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[kInputWeight]);
  (void)types.emplace("weight", input_args[kInputWeight]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt4, kInt8, kInt32}, primitive->name());

  types.clear();
  MS_EXCEPTION_IF_NULL(input_args[kInputAntiquantScale]);
  TypePtr antiquant_scale_type = input_args[kInputAntiquantScale]->GetType();
  (void)types.emplace("antiquant_scale", antiquant_scale_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputAntiquantOffset]);
  if (!input_args[kInputAntiquantOffset]->GetType()->isa<TypeNone>()) {
    (void)types.emplace("antiquant_offset", input_args[kInputAntiquantOffset]->GetType());
  }

  MS_EXCEPTION_IF_NULL(input_args[kInputBias]);
  if (!input_args[kInputBias]->GetType()->isa<TypeNone>()) {
    TypePtr bias_type = input_args[kInputBias]->GetType();
    (void)types.emplace("bias", bias_type);
  }

  MS_EXCEPTION_IF_NULL(input_args[kInputX]);
  MS_EXCEPTION_IF_NULL(input_args[kInputX]->GetType()->cast<TensorTypePtr>());
  auto input_type = input_args[kInputX]->GetType()->cast<TensorTypePtr>()->element();
  if (input_type->type_id() == TypeId::kNumberTypeFloat16) {
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat16}, primitive->name());
  } else if (input_type->type_id() == TypeId::kNumberTypeBFloat16) {
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kBFloat16}, primitive->name());
  } else {
    MS_EXCEPTION(TypeError) << "WeightQuantBatchMatmul inputx type only support f16/bf16, but get " << input_type;
  }

  MS_EXCEPTION_IF_NULL(input_args[kInputQuantScale]);
  if (input_args[kInputQuantScale]->GetType()->isa<TypeNone>()) {
    MS_LOG(INFO) << "WeightQuantBatchMatmulFuncImpl InferType is " << input_args[kInputX]->GetType();
    return input_args[kInputX]->GetType();
  } else {
    types.clear();
    MS_EXCEPTION_IF_NULL(input_args[kInputQuantScale]);
    TypePtr quant_scale_type = input_args[kInputQuantScale]->GetType();
    (void)types.emplace("quant_scale", quant_scale_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kUInt64, kFloat}, primitive->name());

    MS_EXCEPTION_IF_NULL(input_args[kInputQuantOffset]);
    if (!input_args[kInputQuantOffset]->GetType()->isa<TypeNone>()) {
      types.clear();
      (void)types.emplace("quant_offset", input_args[kInputQuantOffset]->GetType());
      (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat}, primitive->name());
    }

    MS_LOG(INFO) << "WeightQuantBatchMatmulFuncImpl InferType is kUInt8";
    return kInt8;
  }
}

void WeightQuantBatchMatmulFuncImpl::CheckBatchMatmulInputWhetherCanBeMul(const std::string &name,
                                                                          const ShapeVector &x_shape,
                                                                          const ShapeVector &weight_shape,
                                                                          bool transpose_x, bool transpose_weight,
                                                                          int type_id) const {
  ShapeVector x_mat_shape(x_shape.end() - SizeToLong(kMatSize), x_shape.end());
  ShapeVector weight_mat_shape(weight_shape.end() - SizeToLong(kMatSize), weight_shape.end());
  int64_t x_col = x_mat_shape[static_cast<size_t>(!transpose_x)];
  int64_t weight_row = weight_mat_shape[static_cast<size_t>(transpose_weight)];
  if (type_id == TypeId::kNumberTypeInt4) {
    weight_row = transpose_weight ? weight_row * 2 : weight_row;  // int4x2 = 2 * int4
  }
  if (std::find(x_shape.begin(), x_shape.end(), -1) == x_shape.end() &&
      std::find(weight_shape.begin(), weight_shape.end(), -1) == weight_shape.end()) {
    if (x_col != weight_row) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                               << x_shape << "(transpose_x=" << transpose_x << "), y shape " << weight_shape
                               << "(transpose_weight=" << transpose_weight << ")";
    }
  }
}

void WeightQuantBatchMatmulFuncImpl::BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp,
                                                          const ShapeVector yshp, bool transpose_x,
                                                          bool transpose_weight, size_t offset) const {
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
  size_t x_offset = xshp.size() - offset;
  size_t weight_offset = yshp.size() - offset;
  output->push_back(xshp[x_offset + (transpose_x ? 1 : 0)]);
  output->push_back(yshp[weight_offset + (transpose_weight ? 0 : 1)]);
  return;
}

void WeightQuantBatchMatmulFuncImpl::CheckBatchMatmulInputSize(const std::string &op_name,
                                                               const std::string &input_name,
                                                               const ShapeVector &shape) const {
  constexpr size_t dim_limit = 2;
  if (shape.size() < dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D or higher dimensional Tensor, but got " << shape.size() << "D shape "
                             << shape;
  }
}

void WeightQuantBatchMatmulFuncImpl::CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name,
                                                                                const ShapeVector &x_shape,
                                                                                const ShapeVector &weight_shape) const {
  ShapeVector x_batch(x_shape.begin(), x_shape.end() - SizeToLong(kMatSize));
  ShapeVector weight_batch(weight_shape.begin(), weight_shape.end() - SizeToLong(kMatSize));
  if (x_batch == weight_batch) {
    return;
  }

  size_t min_size = std::min(x_batch.size(), weight_batch.size());
  for (int64_t i = 0; i < SizeToLong(min_size); ++i) {
    auto x = *(x_batch.rbegin() + i);
    auto y = *(weight_batch.rbegin() + i);
    if (x != 1 && y != 1 && x != y) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", one of the input's batch dim must be equal to another input's peer batch dim, or "
                                  "be equal to 1, or be empty, but got "
                               << x << " and " << y << ", with x shape " << x_shape << ", y shape " << weight_shape;
    }
  }
}

TypePtrList WeightQuantBatchMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputX]->cast<tensor::BaseTensorPtr>();
  const auto &w_tensor = input_values[kInputWeight]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(w_tensor);
  TypePtr x_type = x_tensor->Dtype();

  if (input_values[kInputQuantScale] == mindspore::kNone) {
    MS_LOG(INFO) << "WeightQuantBatchMatmulFuncImpl InferType is " << x_type;
    return {x_type};
  } else {
    MS_LOG(INFO) << "WeightQuantBatchMatmulFuncImpl InferType is kInt8";
    return {kInt8};
  }
}

ShapeArray WeightQuantBatchMatmulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &x_shp = x_tensor->shape();
  const auto &weight_shp = y_tensor->shape();

  auto prim_name = primitive->name();

  auto transpose_x_any = GetScalarValue<bool>(input_values[kInputTransposeX]);

  bool transpose_x = transpose_x_any.value();

  auto transpose_weight_any = GetScalarValue<bool>(input_values[kInputTransposeWeight]);
  bool transpose_weight = transpose_weight_any.value();

  CheckBatchMatmulInputSize(prim_name, "x", x_shp);

  CheckBatchMatmulInputSize(prim_name, "weight", weight_shp);

  CheckBatchMatmulInputWhetherCanBeMul(prim_name, x_shp, weight_shp, transpose_x, transpose_weight,
                                       y_tensor->data_type());

  CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x_shp, weight_shp);

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, weight_shp, transpose_x, transpose_weight, kMatSize);
  if (y_tensor->data_type() == TypeId::kNumberTypeInt4 && !transpose_weight) {
    ret_shape[ret_shape.size() - 1] *= kInt4ShapeMul;
  }
  return {ret_shape};
}
// REGISTER_SIMPLE_INFER(kNameWeightQuantBatchMatmul, WeightQuantBatchMatmulFuncImpl)
}  // namespace ops
}  // namespace mindspore
