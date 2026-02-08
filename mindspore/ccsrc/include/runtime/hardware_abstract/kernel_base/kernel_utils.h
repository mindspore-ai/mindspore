/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_KERNEL_COMMON_OPS_UTILS_H_
#define MINDSPORE_OPS_KERNEL_COMMON_OPS_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace kernel {
RUNTIME_HARDWARE_EXPORT float Scaling(size_t in_size, size_t out_size, bool align_corners);
inline float Scaler(const size_t x, const float scale, bool half_pixel_centers) {
  if (half_pixel_centers) {
    /**
     * function with a std::floor(), so instead of subtracting the 0.5 as we
     * do in HalfPixelScale, we leave it as is, as the std::floor does the
     * correct thing.
     * */
    return (static_cast<float>(x) + 0.5f) * scale;
  } else {
    /**
     * Older incorrect scaling method that causes all resizes to have a slight
     * translation leading to inconsistent results. For example, a flip then a
     * resize gives different results then a resize then a flip.
     * */
    return static_cast<float>(x) * scale;
  }
}

RUNTIME_HARDWARE_EXPORT std::vector<bool> Dec2Bin(const int64_t &mask);
// ===========================New interface==========================================================
RUNTIME_HARDWARE_EXPORT void FillEmptyDims(const std::string &kernel_name, std::vector<int64_t> *begin,
                                           std::vector<int64_t> *end, std::vector<int64_t> *stride,
                                           ShapeVector *input_shape, bool is_gpu_strided = false);
RUNTIME_HARDWARE_EXPORT void ParseStrideSliceMasks(const std::vector<kernel::KernelTensor *> &inputs,
                                                   std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                                   std::vector<int64_t> *stride, const ShapeVector &input_shape);

// ===========================Old interface==========================================================
RUNTIME_HARDWARE_EXPORT void FillEmptyDims(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                           std::vector<int64_t> *end, std::vector<int64_t> *stride,
                                           ShapeVector *input_shape, bool is_gpu_strided = false);

#define CHECK_KERNEL_WORKSPACE_SIZE(actual_size, expect_size, kernel_name)                                           \
  do {                                                                                                               \
    if ((actual_size) != (expect_size)) {                                                                            \
      MS_LOG(EXCEPTION) << (kernel_name) << " requires " << (expect_size) << " workspace, but got " << (actual_size) \
                        << ".";                                                                                      \
    }                                                                                                                \
  } while (0)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_COMMON_OPS_UTILS_H_
