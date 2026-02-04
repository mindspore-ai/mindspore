/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/gcd.h"
#include <set>

#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
static inline bool IsValidGcdType(TypeId t) {
  static const std::set<TypeId> valid_types = {kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  return valid_types.find(t) != valid_types.end();
}

ShapeArray GcdFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_shape = input_infos[0]->GetShape();
  const auto outer_shape = input_infos[1]->GetShape();
  return {CalBroadCastShape(input_shape, outer_shape, primitive->name())};
}

std::vector<TypeId> GcdFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_type_id = input_infos[0]->GetType();
  if (!IsValidGcdType(input_type_id)) {
    MS_EXCEPTION(TypeError)
      << "For Primitive[Gcd], the type of the input tensor must be [Int16, Int32, Int64], but got "
      << TypeIdToString(input_type_id) << "!";
  }
  return {input_type_id};
}

}  // namespace ops
}  // namespace mindspore
