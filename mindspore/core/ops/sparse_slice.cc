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
#include "ops/sparse_slice.h"

#include <memory>
#include <vector>
#include <algorithm>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/math_ops.h"
#include "ops/sparse_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kIndicesRank = 2;
std::vector<abstract::BaseShapePtr> GetOutputShapes(const int64_t max_value_num, const int64_t rank) {
  ShapeVector y_indices_shape{max_value_num, rank};
  ShapeVector y_value_shape{max_value_num};
  ShapeVector y_shape_shape{rank};

  std::vector<abstract::BaseShapePtr> out_shapes{std::make_shared<abstract::TensorShape>(std::move(y_indices_shape)),
                                                 std::make_shared<abstract::TensorShape>(std::move(y_value_shape)),
                                                 std::make_shared<abstract::TensorShape>(std::move(y_shape_shape))};

  return out_shapes;
}
}  // namespace
MIND_API_OPERATOR_IMPL(SparseSlice, BaseOperator);
class SparseSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto indices_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    MS_CHECK_VALUE(indices_shape.size() == kIndicesRank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of indices", SizeToLong(indices_shape.size()),
                                                               kEqual, SizeToLong(kIndicesRank), primitive));

    auto max_value_num = indices_shape[0];
    auto rank = indices_shape[1];
    auto out_shapes = GetOutputShapes(max_value_num, rank);

    return std::make_shared<abstract::TupleShape>(std::move(out_shapes));
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // check shapes
    SparseSliceCheckShape(primitive, input_args);

    // infer shapes
    auto max_value_num = abstract::Shape::kShapeDimAny;
    auto indices_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    auto rank = IsDynamicRank(indices_shape) ? abstract::Shape::kShapeDimAny : indices_shape[kInputIndex1];
    auto out_shapes = GetOutputShapes(max_value_num, rank);

    // get value type
    auto value_tensor_type = input_args[kInputIndex1]->GetType()->cast_ptr<TensorType>();
    MS_EXCEPTION_IF_NULL(value_tensor_type);
    auto value_type = value_tensor_type->element();

    // get outputs
    auto y_indices = std::make_shared<abstract::AbstractTensor>(kInt64, out_shapes[kInputIndex0]);
    auto y_value = std::make_shared<abstract::AbstractTensor>(value_type, out_shapes[kInputIndex1]);
    auto y_shape = std::make_shared<abstract::AbstractTensor>(kInt64, out_shapes[kInputIndex2]);
    AbstractBasePtrList outputs{y_indices, y_value, y_shape};

    return std::make_shared<abstract::AbstractTuple>(std::move(outputs));
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_num = 5;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    auto indices_type = input_args[kInputIndex0]->GetType();
    auto value_type = input_args[kInputIndex1]->GetType();
    auto shape_type = input_args[kInputIndex2]->GetType();
    auto start_type = input_args[kInputIndex3]->GetType();
    auto size_type = input_args[kInputIndex4]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, {kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", shape_type, {kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("start", start_type, {kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, {kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("values", value_type,
                                                     {kUInt8, kUInt16, kUInt32, kUInt64, kInt8, kInt16, kInt32, kInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool},
                                                     op_name);
    auto y_indices_type = std::make_shared<TensorType>(kInt64);
    auto y_shape_type = std::make_shared<TensorType>(kInt64);
    return std::make_shared<Tuple>(std::vector<TypePtr>{y_indices_type, value_type, y_shape_type});
  }

 private:
  static void SparseSliceCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
    auto op_name = primitive->name();
    const auto &indices_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    const auto &values_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
    const auto &shape_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
    const auto &start_shape = input_args[kInputIndex3]->GetShape()->GetShapeVector();
    const auto &size_shape = input_args[kInputIndex4]->GetShape()->GetShapeVector();

    if (IsDynamic(indices_shape) || IsDynamic(values_shape) || IsDynamic(shape_shape) || IsDynamic(start_shape) ||
        IsDynamic(size_shape)) {
      return;
    }

    const int64_t indices_rank = 2;
    (void)CheckAndConvertUtils::CheckInteger("rank of indices", SizeToLong(indices_shape.size()), kEqual, indices_rank,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of values", SizeToLong(values_shape.size()), kEqual, 1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of shape", SizeToLong(shape_shape.size()), kEqual, 1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of start", SizeToLong(start_shape.size()), kEqual, 1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of size", SizeToLong(size_shape.size()), kEqual, 1, op_name);

    if (indices_shape[0] != values_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSlice, indices.shape[0] must equal to values.shape[0], but got indices.shape = " << indices_shape
        << " and value.shape = " << values_shape << ".";
    }
    if (indices_shape[1] != shape_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSlice, indices.shape[1] must equal to shape.shape[0], but got indices.shape = " << indices_shape
        << " and shape.shape = " << shape_shape << ".";
    }
    if (shape_shape != start_shape) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, shape.shape must equal to start.shape, but got shape.shape = "
                               << shape_shape << " and start.shape = " << start_shape << ".";
    }
    if (shape_shape != size_shape) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, shape.shape must equal to size.shape, but got shape.shape = "
                               << shape_shape << "and size.shape = " << size_shape << ".";
    }
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSlice, prim::kPrimSparseSlice, SparseSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
