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

#include "include/pynative/utils/pyboost/functions/composite/randn_like.h"
#include "include/pynative/utils/pyboost/functions/composite/empty_like.h"
#include "include/pynative/utils/pyboost/functions/composite/composite_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
bool IsRandnLikeFloatType(TypeId output_type) {
  switch (output_type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat16:
    case kNumberTypeBFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeFloat64:
      return true;
    default:
      return false;
  }
}
}  // namespace

tensor::TensorPtr randn_like(const tensor::TensorPtr &input, const tensor::TensorPtr &seed,
                             const tensor::TensorPtr &offset, const std::optional<Int64ImmPtr> &dtype,
                             const std::optional<Int64ImmPtr> &device) {
  auto output_type = GetDataType(input, dtype);
  if (!IsRandnLikeFloatType(output_type)) {
    MS_EXCEPTION(TypeError) << "For 'randn_like_ext', the 'dtype' must be a float type, but got "
                            << TypeIdToString(output_type) << ".";
  }
  tensor::TensorPtr output_empty = empty_like(input, dtype, device, std::make_shared<BoolImm>(false));
  if (output_empty->device_address() != nullptr &&
      output_empty->device_address()->GetDeviceType() == device::DeviceType::kMeta) {
    return output_empty;
  }
  return inplace_normal(output_empty, std::make_shared<FP32Imm>(0.), std::make_shared<FP32Imm>(1.), seed, offset);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
