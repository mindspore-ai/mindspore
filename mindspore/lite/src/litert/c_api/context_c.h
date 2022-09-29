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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_C_API_CONTEXT_C_H_
#define MINDSPORE_LITE_SRC_RUNTIME_C_API_CONTEXT_C_H_

#include <string>
#include <vector>
#include <memory>
#include "include/c_api/types_c.h"

namespace mindspore {
class Allocator;
class Delegate;

typedef struct DeviceInfoC {
  MSDeviceType device_type;
  bool enable_fp16 = false;
  int frequency = 3;
  std::string provider;
  std::string provider_device;
  std::shared_ptr<Allocator> allocator = nullptr;
} DeviceInfoC;

typedef struct ContextC {
  std::vector<std::shared_ptr<DeviceInfoC>> device_info_list;
  int32_t thread_num = 2;
  bool enable_parallel = false;
  std::vector<int32_t> affinity_core_list;
  int affinity_mode = 0;
  int delegate_mode = 0;
  std::shared_ptr<Delegate> delegate = nullptr;
} ContextC;
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_C_API_CONTEXT_C_H_
