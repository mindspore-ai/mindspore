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
#ifndef AICPU_KERNELS_NORMALIZED_RESIZE_BICUBIC_H_
#define AICPU_KERNELS_NORMALIZED_RESIZE_BICUBIC_H_

#include <string>

#include "Eigen/Core"
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
struct ResizerState {
  void CalculateSize(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape, bool align_corners_flag);
  int64_t batch_size;
  int64_t out_height;
  int64_t out_width;
  int64_t in_height;
  int64_t in_width;
  int64_t channels;
  float height_scale;
  float width_scale;
  int64_t out_hw_size;
  int64_t in_hw_size;
  int64_t bchw_size;
};

template <typename T1, typename T2>
uint32_t DoCompute(CpuKernelContext &ctx);
class ResizeBicubicCpuKernel : public CpuKernel {
 public:
  ~ResizeBicubicCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);

  template <typename T1, typename T2>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T1, typename T2>
  inline uint32_t InterpolateWithCache(CpuKernelContext &ctx, const T1 *input_data, T2 *output_data);

  DataType dtype_ = DT_INT32;
  ResizerState state_info_;
  bool half_pixel_centers_ = false;
  bool align_corners_ = false;
};
}  // namespace aicpu
#endif
