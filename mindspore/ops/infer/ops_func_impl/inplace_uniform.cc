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
#include "infer/ops_func_impl/inplace_uniform.h"
#include <utility>
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

ShapeArray InplaceUniformFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> InplaceUniformFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kInputIndex0]->GetType();
  auto prim_name = primitive->name();
  const std::set<TypeId> input_valid_types = {
    kNumberTypeUInt8,   kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,    kNumberTypeInt64,
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16, kNumberTypeBool};
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", input_type, input_valid_types, prim_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
