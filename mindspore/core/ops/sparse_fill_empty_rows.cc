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
#include "ops/sparse_fill_empty_rows.h"

#include <set>
#include <string>
#include <map>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
bool CheckSparseFillEmptyRowsInputs(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name) {
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto default_value_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (IsDynamic(indices_shape) || IsDynamic(values_shape) || IsDynamic(dense_shape_shape) ||
      IsDynamic(default_value_shape)) {
    return false;
  }

  const int64_t indice_rank = 2;
  const int64_t values_rank = 1;
  const int64_t dense_shape_rank = 1;
  const int64_t default_value_rank = 0;
  const int64_t dense_rank = 2;

  (void)CheckAndConvertUtils::CheckInteger("indices rank", SizeToLong(indices_shape.size()), kEqual, indice_rank,
                                           op_name);
  if (indices_shape[1] != dense_rank) {
    MS_EXCEPTION(ValueError) << "For SparseFillEmptyRows, "
                             << "the last dim of the indices must be 2, but got " << indices_shape[1] << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("values rank", SizeToLong(values_shape.size()), kEqual, values_rank,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("dense_shape rank", SizeToLong(dense_shape_shape.size()), kEqual,
                                           dense_shape_rank, op_name);
  (void)CheckAndConvertUtils::CheckInteger("dense_shape size", dense_shape_shape[0], kEqual, dense_rank, op_name);
  (void)CheckAndConvertUtils::CheckInteger("default_value rank", SizeToLong(default_value_shape.size()), kEqual,
                                           default_value_rank, op_name);
  if (indices_shape[0] != values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For SparseFillEmptyRows, "
                             << "the size of indices must be equal to values, but got " << indices_shape[0] << " and "
                             << values_shape[0] << ".";
  }
  return true;
}

abstract::TupleShapePtr SparseFillEmptyRowsInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  abstract::ShapePtr output_indices_shape;
  abstract::ShapePtr output_values_shape;
  abstract::ShapePtr output_empty_row_indicator_shape;
  abstract::ShapePtr output_reverse_index_map_shape;

  const int64_t rank = 2;
  auto input_shape_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value);

  if (CheckSparseFillEmptyRowsInputs(input_args, op_name) && !input_shape_value->isa<AnyValue>() &&
      !input_shape_value->isa<None>()) {
    auto indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    const int64_t input_nnz = indice_shape[0];

    auto dense_row = CheckAndConvertUtils::CheckTensorIntValue("x_dense_shape", input_shape_value, op_name)[0];
    output_indices_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny, rank}),
                                                             ShapeVector({input_nnz + dense_row, rank}));
    output_values_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}),
                                                            ShapeVector({input_nnz + dense_row}));
    output_empty_row_indicator_shape = std::make_shared<abstract::Shape>(ShapeVector({dense_row}));
    output_reverse_index_map_shape = std::make_shared<abstract::Shape>(ShapeVector({input_nnz}));
  } else {
    output_indices_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny, rank}));
    output_values_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
    output_empty_row_indicator_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
    output_reverse_index_map_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    output_indices_shape, output_values_shape, output_empty_row_indicator_shape, output_reverse_index_map_shape});
}

TypePtr SparseFillEmptyRowsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> common_valid_types_with_bool_and_complex = {
    kInt8,   kInt16,   kInt32,   kInt64,   kUInt8, kUInt16,    kUInt32,
    kUInt64, kFloat16, kFloat32, kFloat64, kBool,  kComplex64, kComplex128};
  auto indices_type = input_args[kInputIndex0]->BuildType();
  auto values_type = input_args[kInputIndex1]->BuildType();
  auto dense_shape_type = input_args[kInputIndex2]->BuildType();
  auto default_value_type = input_args[kInputIndex3]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("values", values_type);
  (void)types.emplace("default_value", default_value_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_bool_and_complex, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dense_shape", dense_shape_type, {kInt64}, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(kInt64), values_type,
                                                      std::make_shared<TensorType>(kBool),
                                                      std::make_shared<TensorType>(kInt64)});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseFillEmptyRows, BaseOperator);
AbstractBasePtr SparseFillEmptyRowsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = SparseFillEmptyRowsInferType(primitive, input_args);
  auto infer_shape = SparseFillEmptyRowsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSparseFillEmptyRowsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseFillEmptyRows, prim::kPrimSparseFillEmptyRows, AGSparseFillEmptyRowsInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
