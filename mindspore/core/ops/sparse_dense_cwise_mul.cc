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
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "ops/sparse_dense_cwise_mul.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseDenseCwiseMulInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indices_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  auto values_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(values_shape_ptr);
  auto shape_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_shape_ptr);
  auto dense_shape_ptr = input_args[kInputIndex3]->BuildShape();
  MS_EXCEPTION_IF_NULL(dense_shape_ptr);
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto dense_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  const size_t indices_dims = 2;
  if (indices_shape.size() != indices_dims) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "',  the dim of indices must be 2, but got "
                             << indices_shape.size();
  }
  if (values_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "',  the dim of values must be 1, but got "
                             << values_shape.size();
  }
  if (shape_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "',  the dim of shape must be 1, but got "
                             << shape_shape.size();
  }
  if (indices_shape[0] != values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "',  the num of indices  must be equal to the number of value, but got "
                             << indices_shape[0] << " vs " << values_shape[0] << ".";
  }
  if (indices_shape[1] != shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "',  shape[1] of `x1_indices` must be equal to shape[0] of `x1_shape`, but got "
                             << indices_shape[1] << " vs " << shape_shape[0] << ".";
  }
  if (dense_shape.size() > size_t(shape_shape[0])) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "',  the dims of `x2` should be less or equal to the shape[0] of `x1_shape`, but got "
                             << dense_shape.size() << " vs " << shape_shape[0] << ".";
  }
  auto output_shape = input_args[kInputIndex1]->BuildShape()->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr SparseDenseCwiseMulInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex0]->BuildType();
  auto shape_type_ptr = input_args[kInputIndex2]->BuildType();
  std::set<TypePtr> type_set = {kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indiecs_type_ptr, type_set, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", shape_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("values", input_args[kInputIndex1]->BuildType());
  (void)type_dict.emplace("shape", input_args[kInputIndex3]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types_with_complex, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseDenseCwiseMul, BaseOperator);
AbstractBasePtr SparseDenseCwiseMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = SparseDenseCwiseMulInferType(primitive, input_args);
  auto infer_shape = SparseDenseCwiseMulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseDenseCwiseMul, prim::kPrimSparseDenseCwiseMul, SparseDenseCwiseMulInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
