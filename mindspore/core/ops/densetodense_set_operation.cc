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

#include "ops/densetodense_set_operation.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr DenseToDenseSetOperationInferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x1_shape = input_args[0]->BuildShape();
  auto x2_shape = input_args[1]->BuildShape();
  int64_t output_rank_dim = 0;
  auto x1_shape_vec = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1_shape)[kShape];
  auto x1_rank = SizeToLong(x1_shape_vec.size());
  auto x2_shape_vec = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_shape)[kShape];
  auto x2_rank = SizeToLong(x2_shape_vec.size());
  constexpr int64_t kNum2 = 2;
  bool x1_is_dynamic_rank = IsDynamicRank(x1_shape_vec);
  bool x2_is_dynamic_rank = IsDynamicRank(x2_shape_vec);
  if (!x1_is_dynamic_rank) {
    (void)CheckAndConvertUtils::CheckInteger("x1_rank", x1_rank, kGreaterEqual, kNum2, prim_name);
  }
  if (!x2_is_dynamic_rank) {
    (void)CheckAndConvertUtils::CheckInteger("x2_rank", x2_rank, kGreaterEqual, kNum2, prim_name);
  }
  if (!x1_is_dynamic_rank && !x2_is_dynamic_rank && x1_rank != x2_rank) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the rank of input `x1` and `x2` must be equal, "
                             << "but got x1_rank " << x1_rank << " and x2_rank " << x2_rank;
  }
  output_rank_dim = x1_rank;
  if (x1_is_dynamic_rank) {
    output_rank_dim = abstract::Shape::kShapeDimAny;
  }
  bool is_input_dynamic = IsDynamic(x1_shape_vec) || IsDynamic(x2_shape_vec);
  int64_t max_num = 0;
  if (!is_input_dynamic) {
    auto x1_num = std::accumulate(x1_shape_vec.begin(), x1_shape_vec.end(), 1, std::multiplies<int64_t>());
    auto x2_num = std::accumulate(x2_shape_vec.begin(), x2_shape_vec.end(), 1, std::multiplies<int64_t>());
    ShapeVector x1_group_shape_vec;
    x1_group_shape_vec.assign(x1_shape_vec.begin(), x1_shape_vec.end() - 1);
    ShapeVector x2_group_shape_vec;
    x2_group_shape_vec.assign(x2_shape_vec.begin(), x2_shape_vec.end() - 1);
    if (x1_group_shape_vec != x2_group_shape_vec) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", the shapes of the first n-1 dimensions of `x1` and `x2` must be "
                                  "equal, "
                               << "but got x1_shape " << x1_shape->ToString() << " "
                               << "and x2_shape " << x2_shape->ToString();
    }
    std::string set_operation_str = GetValue<std::string>(primitive->GetAttr("set_operation"));
    std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
    if (set_operation_str == "a-b") {
      max_num = x1_num;
    } else if (set_operation_str == "b-a") {
      max_num = x2_num;
    } else if (set_operation_str == "intersection") {
      max_num = std::max(x1_num, x2_num);
    } else if (set_operation_str == "union") {
      max_num = x1_num + x2_num;
    } else {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", the attr set_operation must be any one of "
                                  "['a-b','b-a','intersection','union'], "
                               << "but got " << set_operation_str << ".";
    }
  }

  ShapeVector output_indices_vec = {-1, output_rank_dim};
  ShapeVector max_output_indices_vec;
  ShapeVector min_output_indices_vec;
  if (!is_input_dynamic) {
    max_output_indices_vec = {max_num, output_rank_dim};
    min_output_indices_vec = {0, output_rank_dim};
  }
  ShapeVector output_values_vec = {-1};
  ShapeVector max_output_values_vec;
  ShapeVector min_output_values_vec;
  if (!is_input_dynamic) {
    max_output_values_vec = {max_num};
    min_output_values_vec = {0};
  }
  ShapeVector output_shape_vec = {output_rank_dim};

  std::vector<abstract::BaseShapePtr> shape_tuple;
  abstract::ShapePtr output_indices_shape =
    std::make_shared<abstract::Shape>(output_indices_vec, min_output_indices_vec, max_output_indices_vec);
  abstract::ShapePtr output_values_shape =
    std::make_shared<abstract::Shape>(output_values_vec, min_output_values_vec, max_output_values_vec);
  abstract::ShapePtr output_shape = std::make_shared<abstract::Shape>(output_shape_vec);

  shape_tuple.push_back(output_indices_shape);
  shape_tuple.push_back(output_values_shape);
  shape_tuple.push_back(output_shape);
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TuplePtr DenseToDenseSetOperationInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", input_args[0]->BuildType());
  (void)types.emplace("x2", input_args[1]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(std::make_shared<TensorType>(kInt64));
  type_tuple.push_back(type);
  type_tuple.push_back(std::make_shared<TensorType>(kInt64));
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(DenseToDenseSetOperation, BaseOperator);
AbstractBasePtr DenseToDenseSetOperationInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = DenseToDenseSetOperationInferType(primitive, input_args);
  auto infershape = DenseToDenseSetOperationInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(DenseToDenseSetOperation, prim::kPrimDenseToDenseSetOperation,
                             DenseToDenseSetOperationInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
