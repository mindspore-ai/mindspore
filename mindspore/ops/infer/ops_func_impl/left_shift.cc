/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/left_shift.h"
#include <set>
#include <vector>
#include "abstract/dshape.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray LeftShiftFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_x_shape = input_infos[kIndex0]->GetShape();
  const auto input_y_shape = input_infos[kIndex1]->GetShape();
  return {CalBroadCastShape(input_x_shape, input_y_shape, primitive->name())};
}

std::vector<TypeId> LeftShiftFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  const auto input_x_type = input_infos[kIndex0]->GetType();
  const auto input_y_type = input_infos[kIndex1]->GetType();
  if (input_x_type != input_y_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the dtype of two args should be same, but the first arg dtype "
                            << TypeIdToString(input_x_type) << " are not consistent with second arg dtype "
                            << TypeIdToString(input_y_type);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8,  kNumberTypeInt16,  kNumberTypeInt32,  kNumberTypeInt64,
                                        kNumberTypeUInt8, kNumberTypeUInt16, kNumberTypeUInt32, kNumberTypeUInt64};
  if (valid_types.find(input_x_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the dtype of input should be one of [Int8, Int16, Int32, Int64, UInt8, UInt16, "
                               "UInt32, UInt64], but got "
                            << TypeIdToString(input_x_type) << ".";
  }
  return {input_x_type};
}
}  // namespace ops
}  // namespace mindspore
