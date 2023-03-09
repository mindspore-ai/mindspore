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
#ifndef AICPU_KERNELS_NORMALIZED_IM2COL_H_
#define AICPU_KERNELS_NORMALIZED_IM2COL_H_

#include <string>
#include <vector>

#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
constexpr uint32_t kValue4 = 4;
class Im2colCpuKernel : public CpuKernel {
 public:
  Im2colCpuKernel() = default;
  ~Im2colCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t Im2colCompute(CpuKernelContext &ctx);
  template <typename T>
  void InnerCompute(int64_t c_col, T *x_ptr, T *y_ptr);
  uint32_t Im2colParamCheck(CpuKernelContext &ctx);

  std::vector<int64_t> ksizes;
  // default value for input attr
  std::vector<int64_t> strides = {1};
  std::vector<int64_t> dilations = {1};
  std::string padding_mode = "CALCULATED";
  std::vector<int64_t> pads = {0};
  const std::vector<std::string> padding_modes = {"SAME", "VALID", "CALCULATED"};

  bool is_NCHW;
  int64_t input_channel;
  int64_t input_height;
  int64_t input_width;
  int64_t out_height;
  int64_t out_width;
  int64_t out_plane;
  int64_t total_block;
  int64_t kernel_height;
  int64_t kernel_width;
  int64_t stride_height;
  int64_t stride_width;
  int64_t dilation_height;
  int64_t dilation_width;
  // pad distance
  int64_t pad_height_top;
  int64_t pad_width_left;
};
}  // namespace aicpu
#endif