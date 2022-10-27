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

#include "ops/index_fill.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <functional>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr IndexFillInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);

  const std::set<TypePtr> valid_data_types = common_valid_types_with_complex_and_bool;
  const std::set<TypePtr> valid_dim_types = {kInt32, kInt64};

  // Input 'dim' can be scalar or tensor.
  auto dim_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("dim", dim_type, valid_dim_types, prim_name);

  // Input 'index' must be a tensor with int32 type.
  auto index_type = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", index_type, {kInt32}, prim_name);

  // Input 'x' must be a tensor.
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_data_types, prim_name);

  // Input 'value' must be a tensor.
  auto value_type = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("value", value_type, valid_data_types, prim_name);

  // Input 'x' and 'value' must have the same types.
  std::map<std::string, TypePtr> args;
  (void)args.emplace("x", x_type);
  (void)args.emplace("value", x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_data_types, prim_name);
  return x_type;
}

abstract::ShapePtr IndexFillInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  // dynamic rank
  auto index_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(index_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // dynamic shape
  if (IsDynamic(x_shape) || IsDynamic(index_shape)) {
    ShapeVector out_shape_dyn;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(out_shape_dyn);
  }

  // Input 'dim' must be a tensor with a value or a scalar.
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    auto dim_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto dim_rank = SizeToLong(dim_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("rank of 'dim'", dim_rank, kEqual, 0, prim_name);
  } else if (!input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', 'dim' must be int or Tensor.";
  }

  // Input 'index' must be a scalar/vector.
  auto index_rank = SizeToLong(index_shape.size());
  CheckAndConvertUtils::CheckInRange("rank of 'index'", index_rank, kIncludeBoth, {0, 1}, prim_name);

  // Input 'value' must be a tensor with a value or a scalar.
  if (input_args[kInputIndex3]->isa<abstract::AbstractTensor>()) {
    auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
    auto value_rank = SizeToLong(value_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("rank of 'value'", value_rank, kEqual, 0, prim_name);
  }

  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(IndexFill, BaseOperator);
AbstractBasePtr IndexFillInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  auto dtype = IndexFillInferType(primitive, input_args);
  auto shape = IndexFillInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, dtype);
}

REGISTER_PRIMITIVE_EVAL_IMPL(IndexFill, prim::kPrimIndexFill, IndexFillInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
