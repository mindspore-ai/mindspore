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
#include "ops/matrix_diag_part_v3.h"
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
abstract::ShapePtr MatrixDiagPartV3InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  const int64_t kNumber1 = 1;
  const int64_t kNumber2 = 2;

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto rank = SizeToLong(x_shape.size());
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("x rank", rank, kGreaterEqual, kNumber2, prim_name);
  }

  auto k_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto k_rank = SizeToLong(k_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("k rank", k_rank, kIncludeBoth, {0, kNumber1}, prim_name);

  auto padding_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto padding_value_rank = SizeToLong(padding_shape.size());
  if (!IsDynamic(padding_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("padding_value rank", padding_value_rank, kEqual, 0, prim_name);
  }

  auto k_val_ptr = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(k_val_ptr);
  if (IsDynamic(x_shape) || IsDynamic(k_shape) || !IsValueKnown(k_val_ptr)) {
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  std::vector<int64_t> out_shape;
  (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end() - kNumber2);

  int64_t row = x_shape[LongToSize(rank - kNumber2)];
  int64_t col = x_shape[LongToSize(rank - 1)];
  auto k_val = CheckAndConvertUtils::CheckTensorIntValue("k", k_val_ptr, prim_name);
  size_t k_val_size = LongToSize(k_val.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("k size", SizeToLong(k_val_size), kIncludeBoth, {kNumber1, kNumber2},
                                              prim_name);
  int64_t max_diag_len = 0;
  if (!(k_val[0] > -row && k_val[0] < col)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of k must be in (-x.shape[-2], x.shape[-1]),"
                             << " meaning the value of k must be in (" << -row << ", " << col << ") in this case"
                             << ", but got " << k_val[0] << ".";
  }
  int64_t kValueZero = 0;
  if (k_val_size == 1 || k_val[0] == k_val[1]) {
    max_diag_len = std::min(row + std::min(k_val[0], kValueZero), col + std::min(-k_val[0], kValueZero));
    out_shape.push_back(max_diag_len);
  } else {
    if (!(k_val[1] > -row && k_val[1] < col)) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of k must be in (-x.shape[-2], x.shape[-1]),"
                               << " meaning the value of k must be in (" << -row << ", " << col << ") in this case"
                               << ", but got " << k_val[1] << ".";
    }
    if (!(k_val[0] <= k_val[1])) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", k[0] can not be greater than k[1].";
    }
    max_diag_len = std::min(row + std::min(k_val[1], kValueZero), col + std::min(-k_val[0], kValueZero));
    out_shape.push_back(k_val[1] - k_val[0] + 1);
    out_shape.push_back(max_diag_len);
  }

  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_value = GetValue<int64_t>(max_length_ptr);

  auto true_value = SizeToLong(SizeOf(out_shape));
  if (true_value > max_value) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", the number of elements of output must be less than max length: " << max_value
                             << ", but got " << true_value
                             << "! The shape of output must be reduced or max_length must be increased.";
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MatrixDiagPartV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto padding_value = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);

  (void)abstract::CheckDtypeSame(prim_name, x, padding_value);

  auto x_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim_name);

  const std::set<TypePtr> valid_type = {kInt32};
  auto k_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(k_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("k", k_type, valid_type, prim_name);

  return x_type;
}
}  // namespace
void MatrixDiagPartV3::Init(const std::string &align) { this->set_align(align); }

void MatrixDiagPartV3::set_align(const std::string &align) { (void)this->AddAttr(kAlign, api::MakeValue(align)); }

std::string MatrixDiagPartV3::get_align() const {
  auto value_ptr = GetAttr(kAlign);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(MatrixDiagPartV3, BaseOperator);
AbstractBasePtr MatrixDiagPartV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto align_ptr = primitive->GetAttr(kAlign);
  MS_EXCEPTION_IF_NULL(align_ptr);
  auto align = GetValue<std::string>(align_ptr);
  (void)CheckAndConvertUtils::CheckString(kAlign, align, {"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"},
                                          primitive->name());
  auto infer_type = MatrixDiagPartV3InferType(primitive, input_args);
  auto infer_shape = MatrixDiagPartV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMatrixDiagPartV3Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagPartV3InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagPartV3InferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatrixDiagPartV3Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixDiagPartV3, prim::kPrimMatrixDiagPartV3, AGMatrixDiagPartV3Infer, false);
}  // namespace ops
}  // namespace mindspore
