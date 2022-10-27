/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ATOMIC_ADD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ATOMIC_ADD_H_

#include <cstdint>

#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
#ifndef _MSC_VER
template <typename T, typename U>
void AtomicAddTask(T *const address, const T val) {
  auto *address_as_ull = reinterpret_cast<U *>(address);
  U old = *address_as_ull;
  U assumed = U(0);
  T desired = T(0);
  do {
    assumed = old;
    T *assumed_t = reinterpret_cast<T *>(&assumed);
    U *desired_u = reinterpret_cast<U *>(&desired);
    desired = *assumed_t + static_cast<T>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed, *desired_u);
  } while (assumed != old);
}
#else
template <typename T, typename U>
void AtomicAddTask(T *const address, const T val) {
  *address = (*address) + val;
}
#endif

template <typename T>
void AtomicAdd(T *const address, const T val) {
  switch (sizeof(T)) {
    case sizeof(int8_t): {
      AtomicAddTask<T, int8_t>(address, val);
      break;
    }
    case sizeof(int16_t): {
      AtomicAddTask<T, int16_t>(address, val);
      break;
    }
    case sizeof(int32_t): {
      AtomicAddTask<T, int32_t>(address, val);
      break;
    }
    case sizeof(int64_t): {
      AtomicAddTask<T, int64_t>(address, val);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Dtype " << typeid(T).name() << " is unsupported.";
  }
}
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ATOMIC_ADD_H_
