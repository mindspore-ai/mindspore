/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/tensor_scatter_arithmetic.h"

#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <sstream>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "mindapi/helper.h"
#include "primitive/array_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {
namespace {
bool CheckUpdatesShape(const std::vector<int64_t> &updates_shape, const std::vector<int64_t> &check_shape) {
  if (std::find(updates_shape.begin(), updates_shape.end(), abstract::TensorShape::kShapeRankAny) !=
        updates_shape.end() ||
      std::find(check_shape.begin(), check_shape.end(), abstract::TensorShape::kShapeRankAny) != check_shape.end()) {
    return true;
  }
  if (updates_shape.size() != check_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < updates_shape.size(); ++i) {
    if (updates_shape[i] == -1 || check_shape[i] == -1) {
      continue;
    }
    if (updates_shape[i] != check_shape[i]) {
      return false;
    }
  }
  return true;
}

void CheckIndicesShape(const std::vector<int64_t> &input_x_shape, const std::vector<int64_t> &indices_shape,
                       bool is_dynamic_rank, const std::string &prim_name) {
  const size_t kMinIndicesRank = 2;
  MS_CHECK_VALUE(indices_shape.size() >= kMinIndicesRank,
                 "For " + prim_name + ", the dimension of 'indices' cannot be less than 2,  but got " +
                   std::to_string(indices_shape.size()));
  auto last_dim = indices_shape.back();
  // Input_x_shape is not dynamic rank
  if (!is_dynamic_rank) {
    MS_CHECK_VALUE(last_dim <= SizeToLong(input_x_shape.size()),
                   "For " + prim_name +
                     ", the last dimension of 'indices' must be less than or equal to the dimension of 'input_x', "
                     "but got the last dimension of 'indices': " +
                     std::to_string(last_dim) +
                     " and the dimension of 'input_x': " + std::to_string(input_x_shape.size()));
  }
}

std::string ConvertShapeVectorToStr(const std::vector<int64_t> &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

ShapeVector build_new_shape(const ShapeVector &input_x_shape, const ShapeVector &updates_shape, size_t last_dim) {
  const auto &input_size = input_x_shape.size();
  ShapeVector new_shape = input_x_shape;
  const auto &updates_start = updates_shape.size() - input_size + last_dim;
  for (size_t i = 0; i < input_size - last_dim; ++i) {
    if (new_shape[last_dim + i] == abstract::TensorShape::kShapeDimAny) {
      new_shape[last_dim + i] = updates_shape[updates_start + i];
    }
  }
  return new_shape;
}
}  // namespace

ShapeArray TensorScatterArithmeticFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  auto &input_x_tensor = input_infos[kInputIndex0];
  MS_EXCEPTION_IF_NULL(input_x_tensor);
  auto &indices_tensor = input_infos[kInputIndex1];
  MS_EXCEPTION_IF_NULL(indices_tensor);
  auto &updates_tensor = input_infos[kInputIndex2];
  MS_EXCEPTION_IF_NULL(updates_tensor);
  auto input_x_shape = input_x_tensor->GetShape();
  auto indices_shape = indices_tensor->GetShape();
  auto updates_shape = updates_tensor->GetShape();
  if (IsDynamicRank(indices_shape)) {
    return {input_x_shape};
  }
  CheckIndicesShape(input_x_shape, indices_shape, IsDynamicRank(input_x_shape), prim_name);
  auto last_dim = indices_shape.back();
  if (IsDynamicRank(updates_shape) || last_dim == abstract::TensorShape::kShapeDimAny) {
    return {input_x_shape};
  } else if (IsDynamicRank(input_x_shape)) {
    indices_shape.pop_back();
    MS_CHECK_VALUE(
      indices_shape.size() <= updates_shape.size(),
      "For " + prim_name +
        ", the length of indices_shape must be less than or equal to updates_shape, but got indices_shape size: " +
        std::to_string(indices_shape.size()) + ", updates_shape size: " + std::to_string(updates_shape.size()) + ".");
    ShapeVector new_updates_shape(updates_shape.begin(), updates_shape.begin() + indices_shape.size());
    MS_CHECK_VALUE(CheckUpdatesShape(indices_shape, new_updates_shape),
                   "For " + prim_name + ", indices_shape must equal to new_updates_shape, but got indices_shape: " +
                     ConvertShapeVectorToStr(indices_shape) +
                     ", new_updates_shape: " + ConvertShapeVectorToStr(new_updates_shape) + ".");
    ShapeVector new_input_x_shape(last_dim, abstract::TensorShape::kShapeDimAny);
    if (indices_shape.size() < updates_shape.size()) {
      new_input_x_shape.insert(new_input_x_shape.end(),
                               updates_shape.end() - (updates_shape.size() - indices_shape.size()),
                               updates_shape.end());
    }
    return {new_input_x_shape};
  }
  indices_shape.pop_back();
  indices_shape.insert(indices_shape.end(), input_x_shape.begin() + last_dim, input_x_shape.end());
  MS_CHECK_VALUE(CheckUpdatesShape(indices_shape, updates_shape),
                 "For " + prim_name + ", indices_shape must equal to updates_shape, but got indices_shape: " +
                   ConvertShapeVectorToStr(indices_shape) +
                   ", updates_shape: " + ConvertShapeVectorToStr(updates_shape) + ".");
  if (last_dim == SizeToLong(input_x_shape.size())) {
    return {input_x_shape};
  }
  return {build_new_shape(input_x_shape, updates_shape, LongToSize(last_dim))};
}

std::vector<TypeId> TensorScatterArithmeticFuncImpl::InferType(const PrimitivePtr &primitive,
                                                               const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex0]);
  auto input_x_type = input_infos[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex1]);
  auto indices_type = input_infos[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex2]);
  auto updates_type = input_infos[kInputIndex2]->GetType();
  std::set<TypeId> indices_type_set = {kNumberTypeInt32, kNumberTypeInt64};
  CheckAndConvertUtils::CheckTypeIdValid("indices type", indices_type, indices_type_set, prim_name);
  if (input_x_type != updates_type) {
    MS_EXCEPTION(TypeError) << "For " + prim_name + ", input_x_type must equal to updates type, but got input_x_type: "
                            << TypeIdToType(input_x_type)->ToString()
                            << ", updates_type: " << TypeIdToType(updates_type)->ToString() << ".";
  }
  // Benchmark not support bool.
  if (input_x_type == kNumberTypeBool) {
    MS_EXCEPTION(TypeError) << "For " + prim_name + ", input_x_type cannot be bool.";
  }
  return {input_x_type};
}

}  // namespace ops
}  // namespace mindspore
