/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "device/cpu/cpu_device_address.h"
#include <vector>
#include "device/convert_tensor_utils.h"

namespace mindspore {
namespace device {
namespace cpu {
bool CPUDeviceAddress::SyncDeviceToHost(const std::vector<int> & /*shape*/, size_t size, TypeId type,
                                        void *host_ptr) const {
  if (type == kNumberTypeFloat16) {
    FloatToHalf(host_ptr, ptr_, size / 2);
  } else if (type == kNumberTypeFloat64) {
    FloatToDouble(host_ptr, ptr_, size / sizeof(double));
  }
  return true;
}

bool CPUDeviceAddress::SyncHostToDevice(const std::vector<int> & /*shape*/, size_t size, TypeId type,
                                        const void *host_ptr) const {
  if (type == kNumberTypeFloat16) {
    HalfToFloat(ptr_, host_ptr, size / 2);
  } else if (type == kNumberTypeFloat64) {
    DoubleToFloat(ptr_, host_ptr, size / sizeof(double));
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
