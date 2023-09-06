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

#ifndef AICPU_KERNELS_NORMALIZED_COL2IM_H_
#define AICPU_KERNELS_NORMALIZED_COL2IM_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class Col2imCpuKernel : public CpuKernel {
 public:
  Col2imCpuKernel() = default;
  ~Col2imCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t Col2imParamCheck(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t Col2imCompute(const CpuKernelContext &ctx);
  template <typename T>
  void InnerCompute(int64_t c_col, int64_t input_offset, int64_t output_offset, T *input_data, T *output_data);

  int64_t output_height, output_width;
  int64_t kernel_height, kernel_width;
  int64_t dilation_height, dilation_width;
  int64_t pad_height, pad_width;
  int64_t stride_height, stride_width;

  int64_t height_col, width_col;

  int64_t channels_col, batch_input_size, batch_output_size;
};
}  // namespace aicpu
#endif
