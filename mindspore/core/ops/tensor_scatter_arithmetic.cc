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

#include "ops/tensor_scatter_arithmetic.h"

#include <map>
#include <set>
#include <string>

#include "ops/tensor_scatter_add.h"
#include "ops/tensor_scatter_sub.h"
#include "ops/tensor_scatter_max.h"
#include "ops/tensor_scatter_min.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TensorScatterArithmeticInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  auto updates_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(updates_shape_ptr);
  if (input_x_shape_ptr->IsDynamic() || indices_shape_ptr->IsDynamic() || updates_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto last_dim = indices_shape.back();
  const size_t kMinIndicesRank = 2;
  if (indices_shape.size() < kMinIndicesRank) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the dimension of 'indices' cannot be less than 2,  but got "
                             << indices_shape.size();
  }
  indices_shape.pop_back();
  if (last_dim < SizeToLong(input_x_shape.size())) {
    indices_shape.insert(indices_shape.end(), input_x_shape.begin() + last_dim, input_x_shape.end());
  }
  if (updates_shape != indices_shape) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", "
                             << "updates_shape = indices_shape[:-1] + x_shape[indices_shape[-1]:], but got x_shape: "
                             << input_x_shape_ptr->ToString() << ", indices_shape: " << indices_shape_ptr->ToString()
                             << ", updates_shape: " << updates_shape_ptr->ToString() << ".";
  }
  auto output_shape = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr TensorScatterArithmeticInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> type_set = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  type_dict.emplace("input_x", input_args[kInputIndex0]->BuildType());
  type_dict.emplace("updates", input_args[kInputIndex2]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types, prim_name);
}
}  // namespace

AbstractBasePtr TensorScatterArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = TensorScatterArithmeticInferType(primitive, input_args);
  auto infer_shape = TensorScatterArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_BASE_IMPL(TensorScatterAdd, PrimitiveC, BaseOperator);
MIND_API_BASE_IMPL(TensorScatterSub, PrimitiveC, BaseOperator);
MIND_API_BASE_IMPL(TensorScatterMax, PrimitiveC, BaseOperator);
MIND_API_BASE_IMPL(TensorScatterMin, PrimitiveC, BaseOperator);

REGISTER_PRIMITIVE_EVAL_IMPL(TensorScatterAdd, prim::kPrimTensorScatterAdd, TensorScatterArithmeticInfer, nullptr,
                             true);
REGISTER_PRIMITIVE_EVAL_IMPL(TensorScatterSub, prim::kPrimTensorScatterSub, TensorScatterArithmeticInfer, nullptr,
                             true);
REGISTER_PRIMITIVE_EVAL_IMPL(TensorScatterMax, prim::kPrimTensorScatterMax, TensorScatterArithmeticInfer, nullptr,
                             true);
REGISTER_PRIMITIVE_EVAL_IMPL(TensorScatterMin, prim::kPrimTensorScatterMin, TensorScatterArithmeticInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
