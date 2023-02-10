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
#include <vector>
#include <memory>

#include "ops/sparse_sparse_minimum.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
const size_t kone = 1;
const size_t ktwo = 2;

TuplePtr SparseSparseMinimumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x1_indices_type = input_args[kInputIndex0]->BuildType();
  auto x1_values_type = input_args[kInputIndex1]->BuildType();
  auto x1_shape_type = input_args[kInputIndex2]->BuildType();
  auto x2_indices_type = input_args[kInputIndex3]->BuildType();
  auto x2_values_type = input_args[kInputIndex4]->BuildType();
  auto x2_shape_type = input_args[kInputIndex5]->BuildType();
  const std::set<TypePtr> common_valid_types = {kFloat32, kFloat16, kInt8,  kInt16,  kUInt16,
                                                kUInt8,   kInt32,   kInt64, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1_values", x1_values_type);
  (void)types.emplace("x2_values", x2_values_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_indices", x1_indices_type, {kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_shape", x1_shape_type, {kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_indices", x2_indices_type, {kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_shape", x2_shape_type, {kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  std::vector<TypePtr> types_list = {input_args[kInputIndex0]->BuildType(), input_args[kInputIndex1]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}

abstract::TupleShapePtr SparseSparseMinimumInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x1_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x1_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x2_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x2_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  if (x1_indices_shape.size() != ktwo || x1_values_shape.size() != kone || x1_shape_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For SparseSparseMinimum, input x1_indices should be a 2-D tensor"
                             << ", but got " << x1_indices_shape.size() << "-D"
                             << ", input x1_values should be a 1-D tensor"
                             << ", but got " << x1_values_shape.size() << "-D"
                             << ", input x1_shape should be a 1-D tensor"
                             << ", but got " << x1_shape_shape.size() << "-D";
  }
  if (x1_indices_shape[0] != x1_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x1_indices.shape[0] and x1_values.shape[0] should be the same"
                             << ", but got x1_indices.shape[0] = " << x1_indices_shape[0]
                             << ", x1_values.shape[0] = " << x1_values_shape[0];
  }
  if (x1_indices_shape[1] != x1_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x1_indices.shape[1] and x1_shape.shape[0] should be the same"
                             << ", but got x1_indices.shape[1] = " << x1_indices_shape[1]
                             << ", x1_shape.shape[0] = " << x1_shape_shape[0];
  }
  if (x2_indices_shape.size() != ktwo || x2_values_shape.size() != kone || x2_shape_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For SparseSparseMinimum, input x2_indices should be a 2-D tensor"
                             << ", but got " << x2_indices_shape.size() << "-D"
                             << ", input x2_values should be a 1-D tensor"
                             << ", but got " << x2_values_shape.size() << "-D"
                             << ", input x2_shape should be a 1-D tensor"
                             << ", but got " << x2_shape_shape.size() << "-D";
  }
  if (x2_indices_shape[0] != x2_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x2_indices.shape[0] and x2_values.shape[0] should be the same"
                             << ", but got x2_indices.shape[0] = " << x2_indices_shape[0]
                             << ", x2_values.shape[0] = " << x2_values_shape[0];
  }
  if (x2_indices_shape[1] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x2_indices.shape[1] and x2_shape.shape[0] should be the same"
                             << ", but got x2_indices.shape[1] = " << x2_indices_shape[1]
                             << ", x2_shape.shape[0] = " << x2_shape_shape[0];
  }
  if (x1_shape_shape[0] != x2_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x1_shape.shape[0] and x2_shape.shape[0] should be the same"
                             << ", but got x1_shape.shape[0] = " << x1_shape_shape[0]
                             << ", x2_shape.shape[0] = " << x2_shape_shape[0];
  }
  ShapeVector y_indices_shape = {-1, x1_shape_shape[0]};
  ShapeVector y_indices_min_shape = {0, x1_shape_shape[0]};
  ShapeVector y_indices_max_shape = {x1_indices_shape[0] + x2_indices_shape[0], x1_shape_shape[0]};
  ShapeVector y_values_shape = {-1};
  ShapeVector y_values_min_shape = {0};
  ShapeVector y_values_max_shape = {x1_indices_shape[0] + x2_indices_shape[0]};
  abstract::ShapePtr y_indices_shape_list =
    std::make_shared<abstract::Shape>(y_indices_shape, y_indices_min_shape, y_indices_max_shape);
  abstract::ShapePtr y_values_shape_list =
    std::make_shared<abstract::Shape>(y_values_shape, y_values_min_shape, y_values_max_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_indices_shape_list, y_values_shape_list});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSparseMinimum, BaseOperator);
AbstractBasePtr SparseSparseMinimumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseSparseMinimumInferType(primitive, input_args);
  auto infer_shape = SparseSparseMinimumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSparseMinimum, prim::kPrimSparseSparseMinimum, SparseSparseMinimumInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
