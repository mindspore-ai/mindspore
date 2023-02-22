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
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "ops/sparse_sparse_arithmetic.h"
#include "ops/sparsesparsemaximum.h"
#include "ops/sparse_sparse_minimum.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void CheckSparseSparseArithmeticInputs(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name) {
  auto x1_indices = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  auto x1_values = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  auto x1_shape = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 2);
  auto x2_indices = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 3);
  auto x2_values = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 4);
  auto x2_shape = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 5);

  auto x1_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1_indices->BuildShape())[kShape];
  auto x1_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1_values->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1_shape->BuildShape())[kShape];
  auto x2_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_indices->BuildShape())[kShape];
  auto x2_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_values->BuildShape())[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_shape->BuildShape())[kShape];

  std::vector<ShapeVector> all_shapes = {x1_indices_shape, x1_values_shape, x1_shape_shape,
                                         x2_indices_shape, x2_values_shape, x2_shape_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);

  if (is_dynamic_rank) {
    return;
  }

  const int64_t indice_size = 2;
  const int64_t values_size = 1;
  const int64_t shape_size = 1;
  (void)CheckAndConvertUtils::CheckInteger("x1_indices rank", SizeToLong(x1_indices_shape.size()), kEqual, indice_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("x2_indices rank", SizeToLong(x2_indices_shape.size()), kEqual, indice_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("x1_values rank", SizeToLong(x1_values_shape.size()), kEqual, values_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("x2_values rank", SizeToLong(x2_values_shape.size()), kEqual, values_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("x1_shape rank", SizeToLong(x1_shape_shape.size()), kEqual, shape_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("x2_shape rank", SizeToLong(x2_shape_shape.size()), kEqual, shape_size,
                                           op_name);

  if (is_dynamic) {
    return;
  }

  if (x1_indices_shape[1] != x1_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', x1_indices.shape[1] and x1_shape.shape[0] must be same, but got "
                             << x1_indices_shape[0] << " and " << x1_shape_shape[0];
  }
  if (x2_indices_shape[1] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', x2_indices.shape[1] and x2_shape.shape[0] must be same, but got "
                             << x2_indices_shape[0] << " and " << x2_shape_shape[0];
  }
  if (x1_indices_shape[0] != x1_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', x1_indices.shape[0] and x1_value.shape[0] must be same, but got "
                             << x1_indices_shape[0] << " and " << x1_values_shape[0];
  }
  if (x2_indices_shape[0] != x2_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', x2_indices.shape[0] and x2_value.shape[0] must be same, but got "
                             << x2_indices_shape[0] << " and " << x2_values_shape[0];
  }
  if (x1_shape_shape[0] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', rank of shapes must be same, but got " << x1_shape_shape[0]
                             << " and " << x2_shape_shape[0];
  }
}

abstract::TupleShapePtr SparseSparseArithmeticInferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckSparseSparseArithmeticInputs(input_args, op_name);
  auto x1_indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x2_indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  ShapeVector out_indice_shape = {-1, -1};
  ShapeVector out_value_shape = {-1};
  ShapeVector max_out_indice_shape = {};
  ShapeVector max_out_value_shape = {};
  abstract::ShapePtr y_indices_shape;
  abstract::ShapePtr y_values_shape;
  if (IsDynamic(x1_indice_shape) || IsDynamic(x2_indice_shape)) {
    max_out_indice_shape.push_back(-1);
    max_out_indice_shape.push_back(-1);
    max_out_value_shape.push_back(-1);
  } else {
    max_out_indice_shape.push_back(x1_indice_shape[0] + x2_indice_shape[0]);
    max_out_indice_shape.push_back(x1_indice_shape[1]);
    out_indice_shape[1] = x1_indice_shape[1];
    max_out_value_shape.push_back(x1_indice_shape[0] + x2_indice_shape[0]);
  }
  y_indices_shape = std::make_shared<abstract::Shape>(out_indice_shape, max_out_indice_shape);
  y_values_shape = std::make_shared<abstract::Shape>(out_value_shape, max_out_value_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_indices_shape, y_values_shape});
}

TuplePtr SparseSparseArithmeticInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_indices", input_args[kInputIndex0]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_shape", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_indices", input_args[kInputIndex3]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_shape", input_args[kInputIndex5]->BuildType(), {kInt64},
                                                   op_name);
  auto x1_values_type = input_args[kInputIndex1]->BuildType();
  auto x2_values_type = input_args[kInputIndex4]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"x1_values", x1_values_type}, {"x2_values", x2_values_type}},
                                                  common_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, x1_values_type});
}
}  // namespace

AbstractBasePtr SparseSparseArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = SparseSparseArithmeticInferType(primitive, input_args);
  auto shapes = SparseSparseArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(SparseSparseMinimum, BaseOperator);
MIND_API_OPERATOR_IMPL(SparseSparseMaximum, BaseOperator);

// AG means auto generated
class MIND_API AGSparseSparseArithmeticInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSparseArithmeticInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSparseArithmeticInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSparseArithmeticInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSparseMinimum, prim::kPrimSparseSparseMinimum, AGSparseSparseArithmeticInfer,
                                 false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSparseMaximum, prim::kPrimSparseSparseMaximum, AGSparseSparseArithmeticInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
