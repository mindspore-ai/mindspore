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

#include <map>
#include <string>
#include <set>
#include "ops/ops_func_impl/bitwise.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BitwiseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           op_name);
  return BroadCastInferShape(op_name, input_args);
}

TypePtr BitwiseFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[0]->GetType();
  auto other_type = input_args[1]->GetType();
  auto out_type = input_type;
  std::map<std::string, TypePtr> types;
  if (input_type->isa<TensorType>()) {
    (void)types.emplace("input", input_type);
  }
  if (other_type->isa<TensorType>()) {
    out_type = other_type;
    (void)types.emplace("other", other_type);
  }
  if (types.empty()) {
    MS_EXCEPTION(TypeError)
      << "Ther primitive[BitwiseAnd]'s input arguments[input, other] as least one is a tensor, but got invalid inputs";
  }
  const std::set<TypePtr> valid_types = {kBool, kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return out_type;
}
}  // namespace ops
}  // namespace mindspore
