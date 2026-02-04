/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/outer.h"

#include <vector>
#include <memory>
#include <set>
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
static inline bool IsValidOuterType(TypeId t) {
  static const std::set<TypeId> valid_types = {
    kNumberTypeBool,  kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
    kNumberTypeUInt8, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};
  return valid_types.find(t) != valid_types.end();
}

ShapeArray OuterFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input = input_infos[0];
  auto &vec2 = input_infos[1];
  const auto input_shape = input->GetShape();
  const auto vec2_shape = vec2->GetShape();

  const size_t kTraceInputRank = 1;
  if (!input->IsDynamicRank() && input_shape.size() != kTraceInputRank) {
    MS_EXCEPTION(ValueError) << "For Primitive[Outer], the rank of the input must be 1, but got " << input_shape.size()
                             << "!";
  } else if (!vec2->IsDynamicRank() && vec2_shape.size() != kTraceInputRank) {
    MS_EXCEPTION(ValueError) << "For Primitive[Outer], the rank of the vec2 must be 1, but got " << vec2_shape.size()
                             << "!";
  }
  ShapeValueDType dim0 = 0;
  ShapeValueDType dim1 = 0;
  if (input->IsDynamicRank()) {
    dim0 = abstract::TensorShape::kShapeDimAny;
  } else {
    dim0 = input_shape[0];
  }
  if (vec2->IsDynamicRank()) {
    dim1 = abstract::TensorShape::kShapeDimAny;
  } else {
    dim1 = vec2_shape[0];
  }
  return {ShapeVector{dim0, dim1}};
}

std::vector<TypeId> OuterFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_type_id = input_infos[0]->GetType();
  if (!IsValidOuterType(input_type_id)) {
    MS_EXCEPTION(TypeError) << "For Primitive[Outer], the type of the input tensor must be [Bool , Uint8, Int8, Int16, "
                               "Int32, Int64, Float16, Float32, Float64, BFloat16], but got "
                            << TypeIdToString(input_type_id) << "!";
  }
  return {input_type_id};
}
}  // namespace mindspore::ops
