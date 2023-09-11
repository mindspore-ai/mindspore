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

#ifndef AICPU_KERNELS_NORMALIZED_AVGPOOL_H_
#define AICPU_KERNELS_NORMALIZED_AVGPOOL_H_
#include <string>
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/inc/cpu_types.h"
namespace aicpu {
struct AvgPoolCalcArgs {
  int64_t batch_size = 0;

  int64_t in_size_c = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;

  int64_t out_size_c = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;

  int64_t stride_n = 1;
  int64_t stride_c = 1;
  int64_t stride_h = 1;
  int64_t stride_w = 1;

  int64_t window_n = 1;
  int64_t window_c = 1;
  int64_t window_h = 1;
  int64_t window_w = 1;

  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t pad_top = 0;
  int64_t pad_bottom = 0;
  int64_t pad_left = 0;
  int64_t pad_right = 0;

  std::string data_format = "NCHW";
  int64_t image_size = 0;
};
class AvgPoolCpuKernel : public CpuKernel {
 public:
  AvgPoolCpuKernel() = default;
  ~AvgPoolCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t AvgPoolParamCheck(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t AvgPoolProcess(const CpuKernelContext &ctx, AvgPoolCalcArgs args);

  template <typename T>
  void RealComputeNCHW(int64_t start, int64_t end, AvgPoolCalcArgs args, T *input0, T *output0);

  template <typename T>
  void RealComputeNHWC(int64_t start, int64_t end, AvgPoolCalcArgs args, T *input0, T *output0);

  template <typename T>
  uint32_t AvgPoolCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_AVGPOOL_H_
