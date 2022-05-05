/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/matrix_diag_v3.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kNumber1 = 1;
const int64_t kNumber2 = 2;
void CheckTrueValueValidAndKValue(const std::vector<AbstractBasePtr> &input_args, int64_t row_val, int64_t col_val,
                                  int64_t additional_value, int64_t max_value, int *k_val, size_t k_val_size) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rank = SizeToLong(x_shape.size());
  int64_t true_value = 1;
  for (int64_t i = 0; i < rank - kNumber2; i++) {
    true_value *= x_shape[i];
  }
  true_value *= additional_value;
  true_value *= (row_val * col_val);
  if (true_value > max_value) {
    MS_EXCEPTION(ValueError) << "For MatrixDiagV3, the number of elements of output must be less than max length: "
                             << max_value << ", but got " << true_value
                             << "! The shape of output must be reduced or max_length must be increased.";
  }
  if (!(k_val[0] > -row_val && k_val[0] < col_val)) {
    MS_EXCEPTION(ValueError) << "For MatrixDiagV3, the value of k must be in (-num_rows, num_cols), "
                             << "meaning the value of k must be in (" << -row_val << ", " << col_val
                             << ") in this case, but got " << k_val[0] << ".";
  }
  if (k_val_size == kNumber2 && k_val[0] != k_val[1]) {
    if (!(k_val[1] > -row_val && k_val[1] < col_val)) {
      MS_EXCEPTION(ValueError) << "For MatrixDiagV3, the value of k must be in (-num_rows, num_cols), "
                               << "meaning the value of k must be in (" << -row_val << ", " << col_val
                               << ") in this case, but got " << k_val[1] << ".";
    }
  }
}
int64_t GetValAndCheckSize(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                           size_t index) {
  // get value of specified input and check its size
  auto prim_name = primitive->name();
  if (input_args[index]->isa<abstract::AbstractTensor>() && input_args[index]->BuildValue()->isa<tensor::Tensor>()) {
    auto abstract_tensor = input_args[index]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(abstract_tensor);
    auto tensor_value_ptr = abstract_tensor->BuildValue();
    MS_EXCEPTION_IF_NULL(tensor_value_ptr);
    auto specified_tensor = tensor_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(specified_tensor);
    size_t tensor_val_size = LongToSize(specified_tensor->DataSize());
    if (index == kInputIndex2) {
      CheckAndConvertUtils::CheckInteger("num_rows size", SizeToLong(tensor_val_size), kEqual, kNumber1, prim_name);
    } else if (index == kInputIndex3) {
      CheckAndConvertUtils::CheckInteger("num_cols size", SizeToLong(tensor_val_size), kEqual, kNumber1, prim_name);
    } else if (index == kInputIndex4) {
      CheckAndConvertUtils::CheckInteger("padding_value size", SizeToLong(tensor_val_size), kEqual, kNumber1,
                                         prim_name);
      return 0;
    }
    auto tensor_ptr = reinterpret_cast<int *>(specified_tensor->data_c());
    int64_t tensor_val = static_cast<int64_t>(*tensor_ptr);
    return tensor_val;
  } else {
    MS_EXCEPTION(TypeError) << "For " << prim_name
                            << ", input k, num_rows, num_cols and padding_value must be const Tensor.";
  }
}
abstract::ShapePtr MatrixDiagV3InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();  // then get shape and check rank
  auto k_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto row_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto col_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto padding_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto k_rank = SizeToLong(k_shape.size());
  auto row_rank = SizeToLong(row_shape.size());
  auto col_rank = SizeToLong(col_shape.size());
  auto padding_value_rank = SizeToLong(padding_shape.size());
  CheckAndConvertUtils::CheckInteger("num_rows rank", row_rank, kEqual, 0, prim_name);
  CheckAndConvertUtils::CheckInteger("num_cols rank", col_rank, kEqual, 0, prim_name);
  CheckAndConvertUtils::CheckInteger("padding_value rank", padding_value_rank, kEqual, 0, prim_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("k rank", k_rank, kIncludeBoth, {0, kNumber1}, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInteger("x rank", rank, kGreaterEqual, kNumber1, prim_name);
  int64_t max_diag_len = x_shape[rank - 1];
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_value = GetValue<int64_t>(max_length_ptr);
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex1]->BuildValue()->isa<tensor::Tensor>()) {
    auto k = input_args[kInputIndex1]->cast<abstract::AbstractTensorPtr>();  // get k value and check its size
    MS_EXCEPTION_IF_NULL(k);
    auto k_value_ptr = k->BuildValue();
    MS_EXCEPTION_IF_NULL(k_value_ptr);
    auto k_tensor = k_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(k_tensor);
    auto k_val = reinterpret_cast<int *>(k_tensor->data_c());
    size_t k_val_size = LongToSize(k_tensor->DataSize());
    CheckAndConvertUtils::CheckInRange<int64_t>("k size", SizeToLong(k_val_size), kIncludeBoth, {kNumber1, kNumber2},
                                                prim_name);
    int64_t row_val = GetValAndCheckSize(primitive, input_args, kInputIndex2);  // get row value and check its size
    int64_t col_val = GetValAndCheckSize(primitive, input_args, kInputIndex3);  // get col value and check its size
    (void)GetValAndCheckSize(primitive, input_args, kInputIndex4);              // check size of padding_value
    std::vector<int64_t> out_shape;                                             // calculate out_shape
    int64_t min_num_rows, min_num_cols;
    int64_t additional_value = 1;
    if (k_val_size == 1 || k_val[0] == k_val[1]) {
      min_num_rows = max_diag_len - std::min(k_val[0], 0);
      min_num_cols = max_diag_len + std::max(k_val[0], 0);
      (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end() - 1);
      additional_value = x_shape[rank - kNumber2];
    } else {
      if (!(k_val[0] <= k_val[1]))
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", k[0] can not be greater than k[1].";
      int64_t num_diags = k_val[1] - k_val[0] + 1;
      CheckAndConvertUtils::CheckInteger("x rank", rank, kGreaterEqual, kNumber2, prim_name);
      if (x_shape[rank - kNumber2] != num_diags)
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", the input x_shape[-2] doesn't match with k value.";
      min_num_rows = max_diag_len - std::min(k_val[1], 0);
      min_num_cols = max_diag_len + std::max(k_val[0], 0);
      (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end() - kNumber2);
    }
    if (row_val != -1 && row_val < min_num_rows)
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of rows is too small.";
    if (col_val != -1 && col_val < min_num_cols)
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of columns is too small.";
    if (row_val == -1 && col_val == -1) {
      row_val = std::max(min_num_rows, min_num_cols);
      col_val = row_val;
    } else if (row_val == -1) {
      row_val = min_num_rows;
    } else if (col_val == -1) {
      col_val = min_num_cols;
    }
    if (!(row_val == min_num_rows || col_val == min_num_cols))
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of rows or columns is not consistent with "
                               << "the specified k and x.";
    CheckTrueValueValidAndKValue(input_args, row_val, col_val, additional_value, max_value, k_val, k_val_size);
    out_shape.push_back(row_val);
    out_shape.push_back(col_val);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    ShapeVector out_shape = {-2};
    ShapeVector infer_shape_min = {0};
    ShapeVector infer_shape_max = {max_value};
    return std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }
}

TypePtr MatrixDiagV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex3);
  auto padding_value = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex4);

  (void)abstract::CheckDtypeSame(prim_name, x, padding_value);

  auto x_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim_name);

  const std::set<TypePtr> valid_type = {kInt32};

  auto k_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(k_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("k", k_type, valid_type, prim_name);

  auto row_type = input_args[kInputIndex2]->BuildType();
  MS_EXCEPTION_IF_NULL(row_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("num_rows", row_type, valid_type, prim_name);

  auto col_type = input_args[kInputIndex3]->BuildType();
  MS_EXCEPTION_IF_NULL(col_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("num_cols", col_type, valid_type, prim_name);

  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatrixDiagV3, BaseOperator);
AbstractBasePtr MatrixDiagV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = MatrixDiagV3InferType(primitive, input_args);
  auto infer_shape = MatrixDiagV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixDiagV3, prim::kPrimMatrixDiagV3, MatrixDiagV3Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
