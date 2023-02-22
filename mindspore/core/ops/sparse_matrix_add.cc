/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "ops/sparse_matrix_add.h"

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>

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
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {
constexpr auto kDenseShape = "dense_shape";
constexpr size_t kSMAInputsNum = 12;
constexpr size_t kADenseShapeIdx = 0;
constexpr size_t kABatchPtrIdx = 1;
constexpr size_t kAIndptrIdx = 2;
constexpr size_t kAIndicesIdx = 3;
constexpr size_t kAValuesIdx = 4;
constexpr size_t kBDenseShapeIdx = 5;
constexpr size_t kBBatchPtrIdx = 6;
constexpr size_t kBIndptrIdx = 7;
constexpr size_t kBIndicesIdx = 8;
constexpr size_t kBValuesIdx = 9;
constexpr size_t kAlphaIndex = 10;
constexpr size_t kBetaIndex = 11;
constexpr int64_t kDefaultRank = 2;
constexpr int64_t kBatchedRank = 3;

abstract::TupleShapePtr SparseMatrixAddInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto a_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kADenseShapeIdx]->BuildShape())[kShape];
  auto a_batch = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kABatchPtrIdx]->BuildShape())[kShape];
  auto a_indptr = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kAIndptrIdx]->BuildShape())[kShape];
  auto a_indices = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kAIndicesIdx]->BuildShape())[kShape];
  auto a_values = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kAValuesIdx]->BuildShape())[kShape];
  auto b_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBDenseShapeIdx]->BuildShape())[kShape];
  auto b_batch = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBBatchPtrIdx]->BuildShape())[kShape];
  auto b_indptr = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBIndptrIdx]->BuildShape())[kShape];
  auto b_indices = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBIndicesIdx]->BuildShape())[kShape];
  auto b_values = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kBValuesIdx]->BuildShape())[kShape];

  if (!IsDynamic(a_dense_shape) && a_dense_shape[0] != kBatchedRank && a_dense_shape[0] != kDefaultRank) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the input dense_shape should have rank 2 or 3, "
                             << "but got " << a_dense_shape[0] << ".";
  }

  const int64_t kDimOne = 1;
  // 1-D dense shape
  (void)CheckAndConvertUtils::CheckInteger("A dense shape", SizeToLong(a_dense_shape.size()), kEqual, kDimOne, op_name);
  (void)CheckAndConvertUtils::CheckInteger("B dense shape", SizeToLong(b_dense_shape.size()), kEqual, kDimOne, op_name);
  // 1-D batch
  (void)CheckAndConvertUtils::CheckInteger("A batch", SizeToLong(a_batch.size()), kEqual, kDimOne, op_name);
  (void)CheckAndConvertUtils::CheckInteger("B batch", SizeToLong(b_batch.size()), kEqual, kDimOne, op_name);
  // 1-D indptr
  (void)CheckAndConvertUtils::CheckInteger("A indptr", SizeToLong(a_indptr.size()), kEqual, kDimOne, op_name);
  (void)CheckAndConvertUtils::CheckInteger("B indptr", SizeToLong(b_indptr.size()), kEqual, kDimOne, op_name);
  // 1-D indices
  (void)CheckAndConvertUtils::CheckInteger("A indices", SizeToLong(a_indices.size()), kEqual, kDimOne, op_name);
  (void)CheckAndConvertUtils::CheckInteger("B indices", SizeToLong(b_indices.size()), kEqual, kDimOne, op_name);
  // 1-D values
  (void)CheckAndConvertUtils::CheckInteger("A values", SizeToLong(a_values.size()), kEqual, kDimOne, op_name);
  (void)CheckAndConvertUtils::CheckInteger("B values", SizeToLong(b_values.size()), kEqual, kDimOne, op_name);
  // indices' shape == values' shape
  std::vector<ShapeVector> check_shapes = {a_indices, a_values, b_indices, b_values};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  if (!is_dynamic && (a_indices != a_values || b_indices != b_values)) {
    MS_EXCEPTION(ValueError) << "Indices and values must have same shape, but get A indices shape " << a_indices
                             << ", A values shape " << a_values << "; B indices shape " << b_indices
                             << ", B values shape " << b_values;
  }
  auto c_dense_shape_ptr = std::make_shared<abstract::Shape>(a_dense_shape);
  auto c_batch_ptr = std::make_shared<abstract::Shape>(a_batch);
  auto c_indptr_ptr = std::make_shared<abstract::Shape>(a_indptr);
  auto c_indices_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1});
  auto c_values_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1});
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{c_dense_shape_ptr, c_batch_ptr, c_indptr_ptr, c_indices_ptr, c_values_ptr});
}

TuplePtr SparseMatrixAddInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto a_values_type = input_args[kAValuesIdx]->BuildType();
  auto b_values_type = input_args[kBValuesIdx]->BuildType();
  auto alpha_type = input_args[kAlphaIndex]->BuildType();
  auto beta_type = input_args[kBetaIndex]->BuildType();
  std::map<std::string, TypePtr> value_type;
  (void)value_type.emplace("a values", a_values_type);
  (void)value_type.emplace("b values", b_values_type);
  (void)value_type.emplace("alpha", alpha_type);
  (void)value_type.emplace("beta", beta_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(value_type, valid_types, op_name);
  auto a_indices_type = input_args[kAIndicesIdx]->BuildType();
  auto a_dense_shape_type = input_args[kADenseShapeIdx]->BuildType();
  auto a_batch_ptr_type = input_args[kABatchPtrIdx]->BuildType();
  auto a_index_ptr_type = input_args[kAIndptrIdx]->BuildType();
  auto b_indices_type = input_args[kBIndicesIdx]->BuildType();
  auto b_dense_shape_type = input_args[kBDenseShapeIdx]->BuildType();
  auto b_batch_ptr_type = input_args[kBBatchPtrIdx]->BuildType();
  auto b_index_ptr_type = input_args[kBIndptrIdx]->BuildType();

  const std::set<TypePtr> int_types = {kInt32, kInt64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("a indices type", a_indices_type);
  (void)types.emplace("b indices type", b_indices_type);
  (void)types.emplace("a batch ptr type", a_batch_ptr_type);
  (void)types.emplace("a index ptr type", a_index_ptr_type);
  (void)types.emplace("b batch ptr type", b_batch_ptr_type);
  (void)types.emplace("b index ptr type", b_index_ptr_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, int_types, op_name);

  (void)types.emplace("a dense shape type", a_dense_shape_type);
  (void)types.emplace("b dense shape type", b_dense_shape_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(
    {{"a dense type", a_dense_shape_type}, {"b dense type", b_dense_shape_type}}, int_types, op_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{a_indices_type, a_indices_type, a_indices_type, a_indices_type, a_values_type});
}
}  // namespace

AbstractBasePtr SparseMatrixAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  const int64_t input_num = 12;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto types = SparseMatrixAddInferType(primitive, input_args);
  auto shapes = SparseMatrixAddInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

void SparseMatrixAdd::set_dense_shape(const std::vector<int64_t> &shape) {
  (void)this->AddAttr(kDenseShape, api::MakeValue(shape));
}

std::vector<int64_t> SparseMatrixAdd::get_dense_shape() const {
  auto value_ptr = GetAttr(kDenseShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void SparseMatrixAdd::Init(const std::vector<int64_t> &csr_a, const std::vector<int64_t> &csr_b) {
  auto op_name = this->name();
  if (csr_a != csr_b) {
    MS_LOG(EXCEPTION) << "For " << op_name << "A shape and B shape must be the same, but got A = " << csr_a
                      << ", and B = " << csr_b << ".";
  }
  this->set_dense_shape(csr_a);
}

MIND_API_OPERATOR_IMPL(SparseMatrixAdd, BaseOperator);

// AG means auto generated
class MIND_API AGSparseMatrixAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixAddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixAddInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixAddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixAdd, prim::kPrimSparseMatrixAdd, AGSparseMatrixAddInfer, false);
}  // namespace ops
}  // namespace mindspore
