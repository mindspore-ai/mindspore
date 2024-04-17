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
#include "ops/ops_func_impl/eltwise_arithmetic.h"
#include <bitset>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
TypePtr EltwiseSpeicalIntegerInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_tensor_type = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor_type);
  auto input_type_id = input_tensor_type->element()->type_id();
  if (input_type_id > kNumberTypeBegin && input_type_id < kNumberTypeEnd) {
    static std::bitset<32> intergral_bitset(
      (1 << (kNumberTypeBool - kNumberTypeBegin)) + (1 << (kNumberTypeUInt8 - kNumberTypeBegin)) +
      (1 << (kNumberTypeInt8 - kNumberTypeBegin)) + (1 << (kNumberTypeInt16 - kNumberTypeBegin)) +
      (1 << (kNumberTypeInt32 - kNumberTypeBegin)) + (1 << (kNumberTypeInt64 - kNumberTypeBegin)));
    return intergral_bitset.test(input_type_id - kNumberTypeBegin) > 0 ? std::make_shared<TensorType>(kFloat32)
                                                                       : input_args[kIndex0]->GetType()->Clone();
  }
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
