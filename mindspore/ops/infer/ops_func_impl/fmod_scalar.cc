/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/fmod_scalar.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace ops {
ShapeArray FmodScalarFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> FmodScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  const std::unordered_set<TypeId> kIntBoolTypes{kNumberTypeInt8,  kNumberTypeInt16, kNumberTypeInt32,
                                                 kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeBool};
  auto IsIntBoolType = [&kIntBoolTypes](TypeId type) { return kIntBoolTypes.find(type) != kIntBoolTypes.end(); };

  auto input_type = input_infos[kIndex0]->GetType();
  auto other_type = input_infos[kIndex1]->GetType();
  TypeId out_type;

  if (IsIntBoolType(input_type) && other_type == kNumberTypeFloat32) {
    out_type = kNumberTypeFloat32;
  } else if (input_type == kNumberTypeBool && IsIntBoolType(other_type)) {
    out_type = kNumberTypeInt64;
  } else {
    out_type = input_type;
  }

  return {out_type};
}
}  // namespace ops
}  // namespace mindspore
