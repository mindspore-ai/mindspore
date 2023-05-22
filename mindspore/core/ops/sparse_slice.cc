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

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SparseSlice, BaseOperator);
class SparseSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (!SparseSliceCheckShape(primitive, input_args)) {
      auto y_indices_shape =
        std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}));
      auto y_value_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
      auto y_shape_shape = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));

      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{y_indices_shape, y_value_shape, y_shape_shape});
    }

    auto indices_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    int64_t nnz = indices_shape[0];
    int64_t rank = indices_shape[1];

    auto y_indices_shape =
      std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny, rank}), ShapeVector({nnz, rank}));
    auto y_value_shape =
      std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}), ShapeVector({nnz}));
    auto y_shape_shape = std::make_shared<abstract::Shape>(ShapeVector({rank}));

    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{y_indices_shape, y_value_shape, y_shape_shape});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_num = 5;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    auto indices_type = input_args[kInputIndex0]->BuildType();
    auto value_type = input_args[kInputIndex1]->BuildType();
    auto shape_type = input_args[kInputIndex2]->BuildType();
    auto start_type = input_args[kInputIndex3]->BuildType();
    auto size_type = input_args[kInputIndex4]->BuildType();
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
  static bool SparseSliceCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
    auto op_name = primitive->name();
    auto indices_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto values_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto shape_shape_ptr = input_args[kInputIndex2]->BuildShape();
    auto start_shape_ptr = input_args[kInputIndex3]->BuildShape();
    auto size_shape_ptr = input_args[kInputIndex4]->BuildShape();

    auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
    auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values_shape_ptr)[kShape];
    auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_shape_ptr)[kShape];
    auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(start_shape_ptr)[kShape];
    auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(size_shape_ptr)[kShape];
    if (IsDynamic(indices_shape) || IsDynamic(values_shape) || IsDynamic(shape_shape) || IsDynamic(start_shape) ||
        IsDynamic(size_shape)) {
      return false;
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
        << "For SparseSlice, indices.shape[0] must equal to values.shape[0], but got indices.shape = "
        << indices_shape_ptr->ToString() << " and value.shape = " << values_shape_ptr->ToString() << ".";
    }
    if (indices_shape[1] != shape_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSlice, indices.shape[1] must equal to shape.shape[0], but got indices.shape = "
        << indices_shape_ptr->ToString() << " and shape.shape = " << shape_shape_ptr->ToString() << ".";
    }
    if (shape_shape != start_shape) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, shape.shape must equal to start.shape, but got shape.shape = "
                               << shape_shape_ptr->ToString() << " and start.shape = " << start_shape_ptr->ToString()
                               << ".";
    }
    if (shape_shape != size_shape) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, shape.shape must equal to size.shape, but got shape.shape = "
                               << shape_shape_ptr->ToString() << "and size.shape = " << size_shape_ptr->ToString()
                               << ".";
    }
    return true;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSlice, prim::kPrimSparseSlice, SparseSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
