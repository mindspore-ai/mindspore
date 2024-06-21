/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/dense.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {

constexpr size_t kDenseIndex0 = 0;
constexpr size_t kDenseIndex1 = 1;
constexpr size_t kDenseIndex2 = 2;

}  // namespace

BaseShapePtr DenseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  constexpr auto kInputNum = 3;
  auto input_len = SizeToLong(input_args.size());
  MS_CHECK_VALUE(
    SizeToLong(input_args.size()) == kInputNum,
    CheckAndConvertUtils::FormatCheckIntegerMsg("input_args number", input_len, kEqual, kInputNum, primitive));
  auto x_shp = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto w_shp = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  ShapeVector ret_shape;
  if (IsDynamicRank(x_shp) || IsDynamicRank(w_shp)) {
    ret_shape.push_back(abstract::Shape::kShapeRankAny);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  const size_t kZero = 0;
  const size_t kOne = 1;
  const size_t kTwo = 2;
  if (w_shp.size() == kOne) {
    const auto kDimW = " if the dim of w is 1.";
    if (x_shp.size() < kOne) {
      MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
    }
    if (x_shp[x_shp.size() - 1] != w_shp[0]) {
      MS_EXCEPTION(ValueError) << "The value of x.shape[-1] should be equal to w.shape[0]" << kDimW;
    }
    if (!input_args[kDenseIndex2]->GetType()->isa<TypeNone>()) {
      auto b_shp = input_args[kDenseIndex2]->GetShape()->GetShapeVector();
      if (b_shp.size() != kZero) {
        MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0" << kDimW;
      }
    }
    ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  const auto kDimW = " if the dim of w is 2.";
  if (w_shp.size() != kTwo) {
    MS_EXCEPTION(ValueError) << "The dim of w should be equal to 1 or 2.";
  }
  if (x_shp.size() < kOne) {
    MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
  }
  if (!input_args[kDenseIndex2]->GetType()->isa<TypeNone>()) {
    auto b_shp = input_args[kDenseIndex2]->GetShape()->GetShapeVector();
    if (b_shp.size() != kZero && b_shp.size() != kOne) {
      MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0 or 1" << kDimW;
    }
  }

  auto x_col = x_shp[x_shp.size() - 1];
  auto w_row = w_shp[1];
  if (x_col != -1 && w_row != -1 && x_col != w_row && x_col >= 0 && w_row >= 0) {
    MS_EXCEPTION(ValueError) << "Dense shape error, got x_col: " << x_col << ", w_row: " << w_row
                             << ". In Dense x_col and w_row should be equal." << kDimW;
  }

  ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
  ret_shape.push_back(w_shp[0]);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr DenseFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  const std::set valid_types = {kUInt8,   kInt8,    kInt16,   kInt32,     kInt64,     kBFloat16,
                                kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kDenseIndex0]->GetType());
  (void)types.emplace("w", input_args[kDenseIndex1]->GetType());
  if (!input_args[kDenseIndex2]->GetType()->isa<TypeNone>()) {
    (void)types.emplace("b", input_args[kDenseIndex2]->GetType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_args[kDenseIndex0]->GetType();
}

TypePtrList DenseFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  TypePtr ret_type = x_tensor->Dtype();
  const auto x_dtype_id = x_tensor->data_type();
  const auto y_dtype_id = y_tensor->data_type();
  if (x_dtype_id != y_dtype_id) {
    MS_EXCEPTION(TypeError) << "For Dense, all dtypes should be the same, but got 'input' with "
                            << "dtype: " << x_dtype_id << " and 'other' with dtype: " << y_dtype_id << ".";
  }
  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &bias_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(bias_tensor);
    const auto bias_dtype_id = bias_tensor->data_type();
    if (x_dtype_id != bias_dtype_id) {
      MS_EXCEPTION(TypeError) << "For Dense, all dtypes should be the same, but got 'input' with "
                              << "dtype: " << x_dtype_id << " and 'bias' with dtype: " << bias_dtype_id << ".";
    }
  }

  return {ret_type};
}

ShapeArray DenseFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &x_shp = x_tensor->shape();
  const auto &w_shp = y_tensor->shape();

  const size_t kZero = 0;
  const size_t kOne = 1;
  const size_t kTwo = 2;

  ShapeVector ret_shape;

  if (w_shp.size() == kOne) {
    const auto kDimW = " if the dim of w is 1.";
    if (x_shp.size() < kOne) {
      MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
    }
    if (x_shp[x_shp.size() - 1] != w_shp[0]) {
      MS_EXCEPTION(ValueError) << "The value of x.shape[-1] should be equal to w.shape[0]" << kDimW;
    }
    if (input_values[kInputIndex2] != mindspore::kNone) {
      const auto &bias_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(bias_tensor);
      const auto b_shp = bias_tensor->shape();
      if (b_shp.size() != kZero) {
        MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0" << kDimW;
      }
    }
    ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
    return {ret_shape};
  }

  const auto kDimW = " if the dim of w is 2.";
  if (w_shp.size() != kTwo) {
    MS_EXCEPTION(ValueError) << "The dim of w should be equal to 1 or 2.";
  }
  if (x_shp.size() < kOne) {
    MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
  }
  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &bias_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(bias_tensor);
    const auto b_shp = bias_tensor->shape();
    if (b_shp.size() != kZero && b_shp.size() != kOne) {
      MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0 or 1" << kDimW;
    }
  }

  auto x_col = x_shp[x_shp.size() - 1];
  auto w_row = w_shp[1];
  if (x_col != -1 && w_row != -1 && x_col != w_row && x_col >= 0 && w_row >= 0) {
    MS_EXCEPTION(ValueError) << "Dense shape error, got x_col: " << x_col << ", w_row: " << w_row
                             << ". In Dense x_col and w_row should be equal." << kDimW;
  }

  ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
  ret_shape.push_back(w_shp[0]);
  return {ret_shape};
}
REGISTER_SIMPLE_INFER(kNameDense, DenseFuncImpl)
}  // namespace ops
}  // namespace mindspore
