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
#include <set>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "ops/sparsesparsemaximum.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
void CheckSparseSparseMaximumInputs(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name) {
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

  if (x1_indices_shape[1] != x1_shape_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For SparseSparseMaximum,  x1_indices.shape[1] and x1_shape.shape[0] must be same, but got "
      << x1_indices_shape[0] << " and " << x1_shape_shape[0];
  }
  if (x2_indices_shape[1] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For SparseSparseMaximum, x2_indices.shape[1] and x2_shape.shape[0] must be same, but got "
      << x2_indices_shape[0] << " and " << x2_shape_shape[0];
  }
  if (x1_indices_shape[0] != x1_values_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For SparseSparseMaximum, x1_indices.shape[0] and x1_value.shape[0] must be same, but got "
      << x1_indices_shape[0] << " and " << x1_values_shape[0];
  }
  if (x2_indices_shape[0] != x2_values_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For SparseSparseMaximum, x2_indices.shape[0] and x2_value.shape[0] must be same, but got "
      << x2_indices_shape[0] << " and " << x2_values_shape[0];
  }
  if (x1_shape_shape[0] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For SparseSparseMaximum, rank of shapes must be same, but got " << x1_shape_shape[0]
                             << " and " << x2_shape_shape[0];
  }
}

abstract::TupleShapePtr SparseSparseMaximumInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckSparseSparseMaximumInputs(input_args, op_name);
  auto x1_indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x2_indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  abstract::ShapePtr y_indices_shape;
  abstract::ShapePtr y_values_shape;
  ShapeVector out_indice_shape = {-1, x1_indice_shape[1]};
  ShapeVector out_value_shape = {-1};
  ShapeVector min_out_indice_shape = {};
  ShapeVector max_out_indice_shape = {};
  ShapeVector min_out_value_shape = {};
  ShapeVector max_out_value_shape = {};

  if (x1_indice_shape[0] > x2_indice_shape[0]) {
    min_out_indice_shape.push_back(x1_indice_shape[0]);
    min_out_value_shape.push_back(x1_indice_shape[0]);
  } else {
    min_out_indice_shape.push_back(x2_indice_shape[0]);
    min_out_value_shape.push_back(x2_indice_shape[0]);
  }
  min_out_indice_shape.push_back(x1_indice_shape[1]);
  max_out_indice_shape.push_back(x1_indice_shape[0] + x2_indice_shape[0]);
  max_out_indice_shape.push_back(x1_indice_shape[1]);
  max_out_value_shape.push_back(x1_indice_shape[0] + x2_indice_shape[0]);

  y_indices_shape = std::make_shared<abstract::Shape>(out_indice_shape, min_out_indice_shape, max_out_indice_shape);
  y_values_shape = std::make_shared<abstract::Shape>(out_value_shape, min_out_value_shape, max_out_value_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_indices_shape, y_values_shape});
}

TuplePtr SparseSparseMaximumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_indices", input_args[kInputIndex0]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_shape", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_indices", input_args[kInputIndex3]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_shape", input_args[kInputIndex5]->BuildType(), {kInt64},
                                                   op_name);
  auto x1_indices_type = input_args[kInputIndex1]->BuildType();
  auto x2_indices_type = input_args[kInputIndex4]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"x1_indices", x1_indices_type}, {"x2_indices", x2_indices_type}},
                                                  common_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, x1_indices_type});
}
}  // namespace
MIND_API_OPERATOR_IMPL(SparseSparseMaximum, BaseOperator);

AbstractBasePtr SparseSparseMaximumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = SparseSparseMaximumInferType(primitive, input_args);
  auto shapes = SparseSparseMaximumInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSparseMaximum, prim::kPrimSparseSparseMaximum, SparseSparseMaximumInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
