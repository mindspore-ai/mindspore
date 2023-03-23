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

#include "ops/sparse_reorder.h"

#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseReorderInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto indices_shape_ptr = input_args[0]->BuildShape();
  auto values_shape_ptr = input_args[1]->BuildShape();
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values_shape_ptr)[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  // Args shape and values must be 1D
  (void)CheckAndConvertUtils::CheckInteger("values dim", SizeToLong(values_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("size dim", SizeToLong(shape_shape.size()), kEqual, 1, prim_name);
  if (IsDynamicRank(indices_shape) || IsDynamicRank(values_shape) || IsDynamicRank(shape_shape)) {
    abstract::ShapePtr output0_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    abstract::ShapePtr output1_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output0_shape, output1_shape});
  }
  // Indices  must be 2D
  const int64_t indices_dims = 2;
  std::vector<ShapeVector> all_shapes = {indices_shape, values_shape, shape_shape};
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);
  if (!is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("indices dim", SizeToLong(indices_shape.size()), kEqual, indices_dims,
                                             prim_name);
    // Indices shape must be equal to the first dimension of var
    (void)CheckAndConvertUtils::CheckInteger("size of values", values_shape[0], kEqual, indices_shape[0], prim_name);
    (void)CheckAndConvertUtils::CheckInteger("size of shape", shape_shape[0], kEqual, indices_shape[1], prim_name);
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{indices_shape_ptr, values_shape_ptr});
}

TuplePtr SparseReorderInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto indices_type = input_args[0]->BuildType();
  auto values_type = input_args[1]->BuildType();
  auto shape_type = input_args[2]->BuildType();
  // Args values must be a scalar type
  const std::set<TypePtr> valid_types_values = {kBool,   kInt8,    kInt16,   kInt32,   kInt64,     kUInt8,
                                                kUInt16, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  const std::set<TypePtr> valid_types_indices = {kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("values", values_type, valid_types_values, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types_indices, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", shape_type, valid_types_indices, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{indices_type, values_type});
}
}  // namespace

AbstractBasePtr SparseReorderInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = SparseReorderInferType(primitive, input_args);
  auto infer_shape = SparseReorderInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(SparseReorder, BaseOperator);

// AG means auto generated
class MIND_API AGSparseReorderInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReorderInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReorderInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReorderInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseReorder, prim::kPrimSparseReorder, AGSparseReorderInfer, false);
}  // namespace ops
}  // namespace mindspore
