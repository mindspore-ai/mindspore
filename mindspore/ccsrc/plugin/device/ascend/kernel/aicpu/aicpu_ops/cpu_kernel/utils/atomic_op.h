/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_AICPU_COMMON_ATOMIC_OP_H_
#define AICPU_OPS_AICPU_COMMON_ATOMIC_OP_H_

#include <cstdint>
#include <complex>
#include "inc/kernel_log.h"

namespace aicpu {
template <typename T, typename S>
void AtomicAddTask(T *const address, const T val) {
  auto *address_as_ull = reinterpret_cast<S *>(address);
  S old = *address_as_ull;
  S assumed;
  T desired;
  T *assumed_t = nullptr;
  S *desired_s = nullptr;
  do {
    assumed = old;
    assumed_t = reinterpret_cast<T *>(&assumed);
    desired_s = reinterpret_cast<S *>(&desired);
    desired = (*assumed_t) + static_cast<T>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed, *desired_s);
  } while (assumed != old);
}

template <typename T>
void AtomicAdd(CpuKernelContext &ctx, T *const address, const T val) {
  switch (sizeof(T)) {
    case sizeof(uint8_t): {
      AtomicAddTask<T, uint8_t>(address, val);
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
      CUST_AICPU_LOGE(ctx, "Unsupported aicpu atomic add format!");
  }
}
template <>
inline void AtomicAdd(CpuKernelContext &ctx, std::complex<double> *const address, const std::complex<double> val) {
  auto double_addr = reinterpret_cast<double *>(address);
  AtomicAdd<double>(ctx, double_addr, std::real(val));
  AtomicAdd<double>(ctx, double_addr + 1, std::imag(val));
  return;
}
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_ATOMIC_OP_H_
