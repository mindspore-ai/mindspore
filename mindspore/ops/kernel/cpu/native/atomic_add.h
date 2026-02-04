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
#ifdef _MSC_VER
#include <windows.h>
#include <intrin.h>
#include <type_traits>
#endif

#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
// Constants for type size checks
constexpr size_t k8BitSize = 1;
constexpr size_t k16BitSize = 2;
constexpr size_t k32BitSize = 4;
constexpr size_t k64BitSize = 8;

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
// For 64-bit integer types, use _InterlockedExchangeAdd64
template <typename T, typename U>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) == k64BitSize>::type AtomicAddTask(T *const address,
                                                                                                   const T val) {
  _InterlockedExchangeAdd64(reinterpret_cast<volatile LONGLONG *>(address), val);
}

// For 8-bit integer types, use _InterlockedExchangeAdd8
template <typename T, typename U>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) == k8BitSize>::type AtomicAddTask(T *const address,
                                                                                                  const T val) {
  _InterlockedExchangeAdd8(reinterpret_cast<volatile char *>(address), val);
}

// For 16-bit integer types, use _InterlockedExchangeAdd16
template <typename T, typename U>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) == k16BitSize>::type AtomicAddTask(T *const address,
                                                                                                   const T val) {
  _InterlockedExchangeAdd16(reinterpret_cast<volatile SHORT *>(address), val);
}

// For 32-bit integer types, use _InterlockedExchangeAdd
template <typename T, typename U>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) == k32BitSize>::type AtomicAddTask(T *const address,
                                                                                                   const T val) {
  _InterlockedExchangeAdd(reinterpret_cast<volatile LONG *>(address), val);
}

// For 8-bit non-integer types, use CAS loop with _InterlockedCompareExchange8
template <typename T, typename U>
typename std::enable_if<!std::is_integral<T>::value && sizeof(T) == k8BitSize>::type AtomicAddTask(T *const address,
                                                                                                   const T val) {
  auto *address_as_uint = reinterpret_cast<U *>(address);
  U old = *address_as_uint;
  U assumed;
  T desired;
  do {
    assumed = old;
    T old_value = *reinterpret_cast<T *>(&assumed);
    desired = old_value + val;
    U desired_uint = *reinterpret_cast<U *>(&desired);
    old = _InterlockedCompareExchange8(reinterpret_cast<volatile char *>(address_as_uint),
                                       static_cast<char>(desired_uint), static_cast<char>(assumed));
  } while (assumed != old);
}

// For 16-bit non-integer types, use CAS loop with _InterlockedCompareExchange16
template <typename T, typename U>
typename std::enable_if<!std::is_integral<T>::value && sizeof(T) == k16BitSize>::type AtomicAddTask(T *const address,
                                                                                                    const T val) {
  auto *address_as_uint = reinterpret_cast<U *>(address);
  U old = *address_as_uint;
  U assumed;
  T desired;
  do {
    assumed = old;
    T old_value = *reinterpret_cast<T *>(&assumed);
    desired = old_value + val;
    U desired_uint = *reinterpret_cast<U *>(&desired);
    old = _InterlockedCompareExchange16(reinterpret_cast<volatile SHORT *>(address_as_uint),
                                        static_cast<SHORT>(desired_uint), static_cast<SHORT>(assumed));
  } while (assumed != old);
}

// For 32-bit non-integer types, use CAS loop with _InterlockedCompareExchange
template <typename T, typename U>
typename std::enable_if<!std::is_integral<T>::value && sizeof(T) == k32BitSize>::type AtomicAddTask(T *const address,
                                                                                                    const T val) {
  auto *address_as_uint = reinterpret_cast<U *>(address);
  U old = *address_as_uint;
  U assumed;
  T desired;
  do {
    assumed = old;
    T old_value = *reinterpret_cast<T *>(&assumed);
    desired = old_value + val;
    U desired_uint = *reinterpret_cast<U *>(&desired);
    old = _InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(address_as_uint),
                                      static_cast<LONG>(desired_uint), static_cast<LONG>(assumed));
  } while (assumed != old);
}

// For 64-bit non-integer types, use CAS loop with _InterlockedCompareExchange64
template <typename T, typename U>
typename std::enable_if<!std::is_integral<T>::value && sizeof(T) == k64BitSize>::type AtomicAddTask(T *const address,
                                                                                                    const T val) {
  auto *address_as_uint = reinterpret_cast<U *>(address);
  U old = *address_as_uint;
  U assumed;
  T desired;
  do {
    assumed = old;
    T old_value = *reinterpret_cast<T *>(&assumed);
    desired = old_value + val;
    U desired_uint = *reinterpret_cast<U *>(&desired);
    old = _InterlockedCompareExchange64(reinterpret_cast<volatile LONGLONG *>(address_as_uint),
                                        static_cast<LONGLONG>(desired_uint), static_cast<LONGLONG>(assumed));
  } while (assumed != old);
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
