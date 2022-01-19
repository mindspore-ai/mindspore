/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/lu_solve_.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

#define LuSolve_for(shape)    \
  do {                        \
    for (auto item : shape) { \
      buffer << item << " ";  \
    }                         \
  } while (0)

#define LuSolve_pop()                \
  do {                               \
    for (size_t i = 0; i < 2; i++) { \
      x_shape.pop_back();            \
      lu_data_shape.pop_back();      \
    }                                \
  } while (0)

#define LuSolve_buffer(x_shape, lu_data_shape)                                                                      \
  do {                                                                                                              \
    LuSolve_pop();                                                                                                  \
    buffer << "For LuSolve x's batch dimension does not match lu_data's batch dimension, x's batch dimension is ["; \
    LuSolve_for(x_shape);                                                                                           \
    buffer << "], lu_data's batch dimension is [";                                                                  \
    LuSolve_for(lu_data_shape);                                                                                     \
    buffer << "], the batch dimensions may have different sizes, ";                                                 \
    buffer << "from right to left, the corresponding dimensions must be equal.";                                    \
  } while (0)

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t kDimNum = 2;
  std::ostringstream buffer;
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto lu_data_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto lu_data_shape = lu_data_shape_map[kShape];
  auto lu_pivots_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape());
  auto lu_pivots_shape = lu_pivots_shape_map[kShape];
  if (lu_data_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " lu_data's dimensions should be greater than or equal to 2.";
  }
  if (x_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " x's dimensions should be greater than or equal to 2.";
  }
  if (lu_pivots_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " lu_pivots's dimensions should be greater than or equal to 1.";
  }
  if (lu_data_shape[lu_data_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " input lu_data should be square matrix "
                             << "while row is " << lu_data_shape[lu_data_shape.size() - kDimNum] << ", col is "
                             << lu_data_shape[lu_data_shape.size() - 1] << ".";
  }
  if (x_shape[x_shape.size() - kDimNum] != lu_data_shape[lu_data_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For " << op_name << " x's col rank is not same as lu_data's col rank. "
                             << "x is " << x_shape[x_shape.size() - kDimNum] << ", lu_data is "
                             << lu_data_shape[lu_data_shape.size() - kDimNum] << ".";
  }
  if (x_shape.size() == lu_data_shape.size()) {
    for (size_t i = 0; i <= x_shape.size() - kDimNum; i++) {
      if (x_shape[i] != lu_data_shape[i]) {
        LuSolve_buffer(x_shape, lu_data_shape);
        MS_EXCEPTION(ValueError) << buffer.str();
      }
    }
  } else if (lu_data_shape.size() > x_shape.size()) {
    for (size_t i = 0; i < x_shape.size() - kDimNum; i++) {
      if (x_shape[i] != lu_data_shape[lu_data_shape.size() - x_shape.size() + i]) {
        LuSolve_buffer(x_shape, lu_data_shape);
        MS_EXCEPTION(ValueError) << buffer.str();
      }
    }
  } else {
    for (size_t i = 0; i < lu_data_shape.size() - kDimNum; i++) {
      if (lu_data_shape[i] != x_shape[x_shape.size() - lu_data_shape.size() + i]) {
        LuSolve_buffer(x_shape, lu_data_shape);
        MS_EXCEPTION(ValueError) << buffer.str();
      }
    }
  }
  if (lu_pivots_shape[lu_pivots_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - 1]) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << " the last dimension of lu_pivots must be equal to the last dimension of lu_data, "
                             << "lu_data is " << lu_data_shape[lu_data_shape.size() - 1] << ", lu_pivots is "
                             << lu_pivots_shape[lu_pivots_shape.size() - 1] << ".";
  }
  for (size_t i = 0; i < lu_pivots_shape.size(); i++) {
    if (lu_data_shape[i] != lu_pivots_shape[i]) {
      x_shape.pop_back();
      x_shape.pop_back();
      lu_pivots_shape.pop_back();
      buffer << "For " << op_name
             << " lu_data's batch dimension does not match lu_pivots's batch dimension, lu_data's batch dimension is [";
      LuSolve_for(x_shape);
      buffer << "], lu_pivots's batch dimension is [";
      LuSolve_for(lu_pivots_shape);
      buffer << "], the size of the dimension and the number of each dimension must be the same.";
      MS_EXCEPTION(ValueError) << buffer.str();
    }
  }
  auto dim_vector = lu_data_shape;
  if (x_shape.size() >= lu_data_shape.size()) {
    return std::make_shared<abstract::Shape>(x_shape);
  } else {
    dim_vector[lu_data_shape.size() - 1] = x_shape[x_shape.size() - 1];
    return std::make_shared<abstract::Shape>(dim_vector);
  }
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kDimNum = 2;
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> type;
  (void)type.emplace("x", input_args[0]->BuildType());
  (void)type.emplace("lu_data", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lu_data", input_args[1]->BuildType(), valid_types, prim->name());
  auto out_type = CheckAndConvertUtils::CheckTensorTypeSame(type, valid_types, prim->name());
  const std::set<TypePtr> valid_lu_pivots_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lu_pivots", input_args[kDimNum]->BuildType(), valid_lu_pivots_types,
                                                   prim->name());
  return out_type;
}
}  // namespace

AbstractBasePtr LuSolveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = InferType(primitive, input_args);
  auto infer_shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(LuSolve, prim::kPrimLuSolve, LuSolveInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
