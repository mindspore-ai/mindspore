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
#include "include/pynative/utils/pyboost/functions/composite/to_other.h"

#include "include/pynative/utils/pyboost/functions/composite/to_base.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr to_other(const mindspore::tensor::TensorPtr &self, const mindspore::tensor::TensorPtr &other,
                           const mindspore::BoolImmPtr &non_blocking, const mindspore::BoolImmPtr &copy) {
  device::DeviceType device_value = other->device_address()->GetDeviceType();
  TypeId dtype_value = other->data_type();
  bool non_blocking_value = GetValue<bool>(non_blocking);
  bool copy_value = GetValue<bool>(copy);

  MS_LOG(DEBUG) << "Start ToOther, input is " << self->ToString()
                << " device:" << device::GetDeviceNameByType(device_value)
                << " dtype:" << TypeIdToType(dtype_value)->ToString() << " non_blocking:" << non_blocking_value
                << " copy:" << copy_value;

  auto output = ToProcessor(self, device_value, dtype_value, non_blocking_value, copy_value)
                  .Contiguous()
                  .Cast()
                  .Device()
                  .Copy()
                  .Get();
  MS_LOG(DEBUG) << "End ToDevice";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
