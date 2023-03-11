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
int64_t GetTensorValue(const AbstractBasePtr &arg, const std::string &prim_name, const std::string &arg_name) {
  if (!arg->isa<abstract::AbstractTensor>() || !arg->BuildValue()->isa<tensor::Tensor>()) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input '" << arg_name << "' must be const Tensor.";
  }
  constexpr int64_t number_one = 1;
  auto abstract_tensor = arg->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(abstract_tensor);
  auto tensor_value_ptr = abstract_tensor->BuildValue();
  MS_EXCEPTION_IF_NULL(tensor_value_ptr);
  auto specified_tensor = tensor_value_ptr->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(specified_tensor);
  int64_t tensor_val_size = SizeToLong(specified_tensor->DataSize());
  MS_EXCEPTION_IF_CHECK_FAIL(tensor_val_size == number_one,
                             prim_name + " infers failed when initializing value of '" + arg_name + "'.");
  auto tensor_ptr = reinterpret_cast<int *>(specified_tensor->data_c());
  int64_t tensor_val = static_cast<int64_t>(*tensor_ptr);
  return tensor_val;
}

ShapeVector GetOutputShape(const std::vector<int64_t> &x_shape, int64_t lower_diag_index, int64_t upper_diag_index,
                           int64_t row_val, int64_t col_val, const std::string &prim_name) {
  ShapeVector out_shape;
  auto x_rank = SizeToLong(x_shape.size());
  constexpr int64_t number_one = 1;
  constexpr int64_t number_two = 2;
  if (lower_diag_index != upper_diag_index) {
    if (lower_diag_index > upper_diag_index) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", k[0] must not be greater than k[1], but got k[0] is "
                               << lower_diag_index << ", k[1] is " << upper_diag_index << ".";
    }
    (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", x_rank, kGreaterEqual, number_two, prim_name);
    auto num_diags = upper_diag_index - lower_diag_index + 1;
    if (x_shape[LongToSize(x_rank - number_two)] != num_diags) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the input x_shape[-2] doesn't match with k value.";
    }
    (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end() - number_two);
  } else {
    (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end() - number_one);
  }

  int64_t max_diag_len = x_shape.back();
  int64_t min_num_rows = max_diag_len - std::min(upper_diag_index, int64_t(0));
  int64_t min_num_cols = max_diag_len + std::max(lower_diag_index, int64_t(0));
  if (row_val != -1 && row_val < min_num_rows) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of rows is too small.";
  }
  if (col_val != -1 && col_val < min_num_cols) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of columns is too small.";
  }
  if (row_val == -1 && col_val == -1) {
    row_val = std::max(min_num_rows, min_num_cols);
    col_val = row_val;
  } else if (row_val == -1) {
    row_val = min_num_rows;
  } else if (col_val == -1) {
    col_val = min_num_cols;
  }
  if (!(row_val == min_num_rows || col_val == min_num_cols)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the number of rows or columns is not consistent with "
                             << "the specified k and x.";
  }
  if (!(lower_diag_index > -row_val && lower_diag_index < col_val)) {
    MS_EXCEPTION(ValueError) << "For MatrixDiagV3, the value of k must be in (-num_rows, num_cols), "
                             << "meaning the value of k must be in (" << -row_val << ", " << col_val
                             << ") in this case, but got " << lower_diag_index << ".";
  }
  if (!(upper_diag_index > -row_val && upper_diag_index < col_val)) {
    MS_EXCEPTION(ValueError) << "For MatrixDiagV3, the value of k must be in (-num_rows, num_cols), "
                             << "meaning the value of k must be in (" << -row_val << ", " << col_val
                             << ") in this case, but got " << upper_diag_index << ".";
  }
  out_shape.push_back(row_val);
  out_shape.push_back(col_val);
  return out_shape;
}

abstract::ShapePtr MatrixDiagV3InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto k_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto row_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto col_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto padding_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  auto k_rank = SizeToLong(k_shape.size());
  auto row_rank = SizeToLong(row_shape.size());
  auto col_rank = SizeToLong(col_shape.size());
  auto padding_value_rank = SizeToLong(padding_shape.size());
  constexpr int64_t number_one = 1;
  constexpr int64_t number_two = 2;
  if (input_args[kInputIndex0]->BuildShape()->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  if (IsDynamicRank(x_shape)) {
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(out_shape);
  }
  CheckAndConvertUtils::CheckInRange<int64_t>("rank of 'k'", k_rank, kIncludeBoth, {0, number_one}, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'num_rows'", row_rank, kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'num_cols'", col_rank, kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'padding_value'", padding_value_rank, kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", x_rank, kGreaterEqual, number_one, prim_name);
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex1]->BuildValue()->isa<tensor::Tensor>()) {
    auto k = input_args[kInputIndex1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(k);
    auto k_value_ptr = k->BuildValue();
    MS_EXCEPTION_IF_NULL(k_value_ptr);
    auto k_tensor = k_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(k_tensor);
    auto k_val = reinterpret_cast<int *>(k_tensor->data_c());
    int64_t k_val_size = SizeToLong(k_tensor->DataSize());
    CheckAndConvertUtils::CheckInRange<int64_t>("size of 'k'", k_val_size, kIncludeBoth, {number_one, number_two},
                                                prim_name);
    int64_t lower_diag_index = static_cast<int64_t>(k_val[0]);
    int64_t upper_diag_index = lower_diag_index;
    if (k_val_size == number_two) {
      upper_diag_index = static_cast<int64_t>(k_val[1]);
    }
    int64_t row_val = GetTensorValue(input_args[kInputIndex2], prim_name, "num_rows");
    int64_t col_val = GetTensorValue(input_args[kInputIndex3], prim_name, "num_cols");
    (void)GetTensorValue(input_args[kInputIndex4], prim_name, "padding_value");

    auto out_shape = GetOutputShape(x_shape, lower_diag_index, upper_diag_index, row_val, col_val, prim_name);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    // Since the real output shape relies on the value of 'k', 'num_cols' and 'num_rows',
    // the out_shape is set to {-2} meaning that even the dimension can not be determined.
    ShapeVector out_shape = {-2};
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr MatrixDiagV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex3);
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

void MatrixDiagV3::Init(const std::string &align) { this->set_align(align); }

void MatrixDiagV3::set_align(const std::string &align) { (void)this->AddAttr(kAlign, api::MakeValue(align)); }

std::string MatrixDiagV3::get_align() const {
  auto value_ptr = GetAttr(kAlign);
  return GetValue<std::string>(value_ptr);
}

AbstractBasePtr MatrixDiagV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t inputs_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, primitive->name());
  // Check 'align' attribute.
  auto align_ptr = primitive->GetAttr(kAlign);
  MS_EXCEPTION_IF_NULL(align_ptr);
  auto align = GetValue<std::string>(align_ptr);
  (void)CheckAndConvertUtils::CheckString(kAlign, align, {"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"},
                                          primitive->name());
  auto infer_type = MatrixDiagV3InferType(primitive, input_args);
  auto infer_shape = MatrixDiagV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MatrixDiagV3, BaseOperator);

// AG means auto generated
class MIND_API AGMatrixDiagV3Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagV3InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagV3InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagV3Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixDiagV3, prim::kPrimMatrixDiagV3, AGMatrixDiagV3Infer, false);
}  // namespace ops
}  // namespace mindspore
