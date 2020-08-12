/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_UTILS_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_UTILS_H_

#include <string>
#include <vector>
#include "CL/cl2.hpp"
#include "utils/log_adapter.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"

namespace mindspore::kernel {

/**
 * GetLocalSize
 * @param number
 * @param max_divider
 * @return
 */
template <typename T, typename N>
T GetBiggestDividerWithPriority(T number, N max_divider) {
  if (number % 8 == 0 && 8 <= max_divider) {
    return (T)8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return (T)4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return (T)2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return (T)i;
    }
  }
  return (T)1;
}

/**
 * GetLocalSize
 * @param n must be non negative
 * @param divisor must be greater than zero
 * @return
 */
template <typename T, typename N>
T DivideRoundUp(T n, N divisor) {
  const T div = static_cast<T>(divisor);
  const T q = n / div;
  return n % div == 0 ? q : q + 1;
}

/**
 * GetLocalSize
 * @param number
 * @param n
 * @return
 */
template <typename T, typename N>
T AlignByN(T number, N n) {
  return DivideRoundUp(number, n) * n;
}

// GetGlobalSize
std::vector<size_t> GetCommonGlobalSize(const std::vector<size_t> &local, const std::vector<size_t> &global);

// GetLocalSize
std::vector<size_t> GetCommonLocalSize(const std::vector<size_t> &global, int max_size);

std::string CLErrorCode(cl_int error_code);

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_UTILS_H_
