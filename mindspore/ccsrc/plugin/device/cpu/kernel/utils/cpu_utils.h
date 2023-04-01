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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_

#include <cmath>

#include "mindspore/core/base/float16.h"

namespace mindspore {
namespace kernel {
// compatible with MSVC
template <typename T>
inline bool IsNan(T x) {
  return std::isnan(x);
}

template <>
inline bool IsNan<float16>(float16 x) {
  return isnan(x);
}

#ifdef _MSC_VER
template <>
inline bool IsNan<int8_t>(int8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint8_t>(uint8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int16_t>(int16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint16_t>(uint16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int32_t>(int32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint32_t>(uint32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int64_t>(int64_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint64_t>(uint64_t x) {
  return isnan(static_cast<double>(x));
}
#endif
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_
