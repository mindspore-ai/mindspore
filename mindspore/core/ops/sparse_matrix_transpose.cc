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

#include <set>
#include <map>
#include <string>

#include "ops/sparse_matrix_transpose.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseMatrixTransposeInferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNoBatch = 2;
  const int64_t kInputWithBatch = 3;
  const int64_t ktwo = 2;
  std::vector<int64_t> dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  const int64_t rank_x = dense_shape_shape[0];
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  const int64_t max_length = GetValue<int64_t>(max_length_ptr);
  std::vector<int64_t> batch_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<int64_t> row_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  std::vector<int64_t> col_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  std::vector<int64_t> values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  if (!IsDynamic(dense_shape_shape) && rank_x != kInputNoBatch && rank_x != kInputWithBatch) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the rank of input must be 2 or 3, but got "
                             << dense_shape_shape.size() << "!";
  }
  if (batch_pointers_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input batch pointers must be 1-D, but got "
                             << batch_pointers_shape.size() << "-D.";
  }
  if (row_pointers_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input row pointers must be 1-D, but got "
                             << row_pointers_shape.size() << "-D.";
  }
  if (col_indices_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input col indices must be 1-D, but got "
                             << col_indices_shape.size() << "-D.";
  }
  if (values_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input col indices must be 1-D, but got "
                             << col_indices_shape.size() << "-D.";
  }
  ShapeVector transpose_row_pointers_shape{abstract::Shape::kShapeDimAny};
  auto dense_shape = input_args[kInputIndex0];
  if (dense_shape->isa<abstract::AbstractTensor>() && dense_shape->BuildValue()->isa<tensor::Tensor>()) {
    auto dense_shape_ = dense_shape->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(dense_shape_);
    auto dense_shape_value = dense_shape->BuildValue();
    MS_EXCEPTION_IF_NULL(dense_shape_value);
    auto dense_shape_tensor = CheckAndConvertUtils::CheckTensorIntValue("dense_shape", dense_shape_value, prim_name);
    if (rank_x == kInputNoBatch) {
      transpose_row_pointers_shape[0] = dense_shape_tensor[1] + 1;
    } else {
      transpose_row_pointers_shape[0] = dense_shape_tensor[0] * (dense_shape_tensor[ktwo] + 1);
    }
    if (transpose_row_pointers_shape[0] > max_length) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << "the shape of output row pointers must be "
                               << "less than max length: " << max_length << ", but got "
                               << transpose_row_pointers_shape[0]
                               << "! The shape of output row pointers should be reduced"
                               << " or max_length should be increased.";
    }
  }
  abstract::ShapePtr dense_shape_shape_ptr = std::make_shared<abstract::Shape>(dense_shape_shape);
  abstract::ShapePtr batch_pointers_shape_ptr = std::make_shared<abstract::Shape>(batch_pointers_shape);
  abstract::ShapePtr transpose_row_pointers_shape_ptr = std::make_shared<abstract::Shape>(transpose_row_pointers_shape);
  abstract::ShapePtr col_indices_shape_ptr = std::make_shared<abstract::Shape>(col_indices_shape);
  abstract::ShapePtr values_shape_ptr = std::make_shared<abstract::Shape>(values_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{dense_shape_shape_ptr, batch_pointers_shape_ptr,
                                        transpose_row_pointers_shape_ptr, col_indices_shape_ptr, values_shape_ptr});
}

TuplePtr SparseMatrixTransposeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> values_valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  auto dense_shape_type = input_args[kInputIndex0]->BuildType();
  auto batch_type = input_args[kInputIndex1]->BuildType();
  auto row_type = input_args[kInputIndex2]->BuildType();
  auto col_type = input_args[kInputIndex3]->BuildType();
  auto value_type = input_args[kInputIndex4]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x_dense_shape", dense_shape_type);
  (void)types.emplace("x_batch_pointers", batch_type);
  (void)types.emplace("x_row_pointers", row_type);
  (void)types.emplace("x_col_indices", col_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, index_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", value_type, values_valid_types, prim->name());
  std::vector<TypePtr> types_list = {input_args[kInputIndex0]->BuildType(), input_args[kInputIndex1]->BuildType(),
                                     input_args[kInputIndex2]->BuildType(), input_args[kInputIndex3]->BuildType(),
                                     input_args[kInputIndex4]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace

void SparseMatrixTranspose::Init(const bool conjugate) { this->set_conjugate(conjugate); }

void SparseMatrixTranspose::set_conjugate(const bool conjugate) {
  (void)this->AddAttr(kConjugate, api::MakeValue(conjugate));
}

bool SparseMatrixTranspose::get_conjugate() const { return GetValue<bool>(GetAttr(kConjugate)); }

AbstractBasePtr SparseMatrixTransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseMatrixTransposeInferType(primitive, input_args);
  auto infer_shape = SparseMatrixTransposeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseMatrixTranspose, BaseOperator);

// AG means auto generated
class MIND_API AGSparseMatrixTransposeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixTransposeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixTransposeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixTransposeInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixTranspose, prim::kPrimSparseMatrixTranspose, AGSparseMatrixTransposeInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
