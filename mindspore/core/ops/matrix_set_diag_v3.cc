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

#include "ops/matrix_set_diag_v3.h"
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
void TrueValueCalAndCheck(const std::vector<AbstractBasePtr> &input_args, int64_t max_value) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rank = SizeToLong(x_shape.size());
  int64_t true_value = 1;
  for (int64_t i = 0; i < rank; i++) {
    true_value *= x_shape[LongToSize(i)];
  }
  if (true_value > max_value) {
    MS_EXCEPTION(ValueError) << "For MatrixSetDiagV3"
                             << ", the number of elements of output must be less than max length: " << max_value
                             << ", but got " << true_value
                             << "! The shape of output must be reduced or max_length must be increased.";
  }
}

abstract::ShapePtr MatrixSetDiagV3InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kNumber2 = 2;
  const int64_t kNumber1 = 1;

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rank = SizeToLong(x_shape.size());

  auto diagonal_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto diagonal_rank = SizeToLong(diagonal_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("diagonal rank", diagonal_rank, kGreaterEqual, kNumber1, prim_name);

  auto k_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto k_rank = SizeToLong(k_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("k rank", k_rank, kIncludeBoth, {0, kNumber1}, prim_name);

  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  (void)CheckAndConvertUtils::CheckInteger("x rank", rank, kGreaterEqual, kNumber2, prim_name);

  std::vector<ShapeVector> shapes = {x_shape, diagonal_shape, k_shape};
  auto is_dynamic = std::any_of(shapes.begin(), shapes.end(), IsDynamic);
  if (is_dynamic) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  auto max_value = GetValue<int64_t>(max_length_ptr);
  TrueValueCalAndCheck(input_args, max_value);

  auto value_ptr = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!IsValueKnown(value_ptr)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  int64_t row = x_shape[LongToSize(rank - kNumber2)];
  int64_t col = x_shape[LongToSize(rank - 1)];

  for (int64_t i = 0; i < rank - kNumber2; i++) {
    if (diagonal_shape[LongToSize(i)] != x_shape[LongToSize(i)]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", diagonal shape value don't match with x shape value.";
    }
  }

  auto k_val = CheckAndConvertUtils::CheckTensorIntValue("k", value_ptr, prim_name);
  auto k_val_size = k_val.size();
  CheckAndConvertUtils::CheckInRange<int64_t>("k size", SizeToLong(k_val_size), kIncludeBoth, {kNumber1, kNumber2},
                                              prim_name);
  int64_t max_diag_len = 0;
  int64_t last_shape_diagonal = diagonal_shape[LongToSize(diagonal_rank - 1)];
  if (!(k_val[0] > -row && k_val[0] < col)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of k must be in (-x.shape[-2], x.shape[-1]),"
                             << " meaning the value of k must be in (" << -row << ", " << col << ") in this case"
                             << ", but got " << k_val[0] << ".";
  }

  if (k_val_size == 1 || k_val[0] == k_val[1]) {
    if (diagonal_rank != rank - 1) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", diagonal rank size don't match with x rank size.";
    }
    max_diag_len = std::min(row + std::min(k_val[0], int64_t(0)), col + std::min(-k_val[0], int64_t(0)));
  } else {
    if (!(k_val[1] > -row && k_val[1] < col)) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of k must be in (-x.shape[-2], x.shape[-1]),"
                               << " meaning the value of k must be in (" << -row << ", " << col << ") in this case"
                               << ", but got " << k_val[1] << ".";
    }
    if (k_val[0] > k_val[1]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", k[0] can not be greater than k[1].";
    }
    if (diagonal_rank != rank) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", diagonal rank size don't match with x rank size.";
    }
    max_diag_len = std::min(row + std::min(k_val[1], int64_t(0)), col + std::min(-k_val[0], int64_t(0)));
    int64_t in_row_diagonal = diagonal_shape[LongToSize(diagonal_rank - kNumber2)];
    int64_t num_diags = IntToLong(k_val[1]) - IntToLong(k_val[0]) + 1;
    if (num_diags != in_row_diagonal) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", diagonal.shape[-2] is not equal to num_diags calculated by k[1] - k[0] + 1, "
                               << "which value is " << num_diags
                               << " in this case, but got diagonal.shape[-2]: " << in_row_diagonal << " in this case.";
    }
  }

  if (max_diag_len != last_shape_diagonal) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", diagonal.shape[-1] is not equal to "
                             << "max_diag_len calculated by min(x.shape[-2] + min(k[1], 0), x.shape[-1] + "
                             << "min(-k[0], 0)), which value is " << max_diag_len
                             << " in this case, but got diagonal.shape[-1]: " << last_shape_diagonal
                             << " in this case.";
  }

  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MatrixSetDiagV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x_arg = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto diagonal_arg = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto k_arg = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  (void)abstract::CheckDtypeSame(prim_name, x_arg, diagonal_arg);
  auto x_type = x_arg->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim_name);
  const std::set<TypePtr> valid_type = {kInt32};
  auto k_type = k_arg->BuildType();
  MS_EXCEPTION_IF_NULL(k_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("k", k_type, valid_type, prim_name);
  return x_type;
}
}  // namespace

void MatrixSetDiagV3::Init(const std::string &align) { this->set_align(align); }

void MatrixSetDiagV3::set_align(const std::string &align) { (void)this->AddAttr(kAlign, api::MakeValue(align)); }

std::string MatrixSetDiagV3::get_align() const {
  auto value_ptr = GetAttr(kAlign);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(MatrixSetDiagV3, BaseOperator);

AbstractBasePtr MatrixSetDiagV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = MatrixSetDiagV3InferType(primitive, input_args);
  auto infer_shape = MatrixSetDiagV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMatrixSetDiagV3Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSetDiagV3InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSetDiagV3InferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixSetDiagV3Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixSetDiagV3, prim::kPrimMatrixSetDiagV3, AGMatrixSetDiagV3Infer, false);
}  // namespace ops
}  // namespace mindspore
