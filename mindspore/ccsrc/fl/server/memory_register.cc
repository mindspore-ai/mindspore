/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "fl/server/memory_register.h"
#include <utility>

namespace mindspore {
namespace fl {
namespace server {
void MemoryRegister::RegisterAddressPtr(const std::string &name, const AddressPtr &address) {
  MS_ERROR_IF_NULL_WO_RET_VAL(address);
  (void)addresses_.try_emplace(name, address);
}

void MemoryRegister::StoreFloatArray(std::unique_ptr<float[]> *array) {
  MS_ERROR_IF_NULL_WO_RET_VAL(array);
  float_arrays_.push_back(std::move(*array));
}

void MemoryRegister::StoreInt32Array(std::unique_ptr<int[]> *array) {
  MS_ERROR_IF_NULL_WO_RET_VAL(array);
  int32_arrays_.push_back(std::move(*array));
}

void MemoryRegister::StoreUint64Array(std::unique_ptr<size_t[]> *array) {
  MS_ERROR_IF_NULL_WO_RET_VAL(array);
  uint64_arrays_.push_back(std::move(*array));
}

void MemoryRegister::StoreCharArray(std::unique_ptr<char[]> *array) {
  MS_ERROR_IF_NULL_WO_RET_VAL(array);
  char_arrays_.push_back(std::move(*array));
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
