/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_GENERATE_EOD_MASK_V2_H_
#define AICPU_KERNELS_GENERATE_EOD_MASK_V2_H_

#include <cstdint>
#include <map>
#include <vector>
#include <functional>
#include "ops/op_enum.h"
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class GenerateEodMaskV2CpuKernel : public CpuKernel {
 public:
  GenerateEodMaskV2CpuKernel() = default;
  ~GenerateEodMaskV2CpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParamCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeKernel(CpuKernelContext &ctx);

  using KernelFunc = std::function<uint32_t(GenerateEodMaskV2CpuKernel *, CpuKernelContext &)>;
  static std::map<DataType, KernelFunc> func_map_;

 private:
  template <typename T>
  uint32_t Memcpy(CpuKernelContext &ctx, const T *input, T *output, size_t num);

  template <typename T>
  uint32_t FaultInjection(CpuKernelContext &ctx, T *output, int64_t num);

 private:
  template <typename T>
  void ModesKernel(CpuKernelContext &ctx, T *output, int64_t num);

  template <typename T>
  void MultiplyMaxElementKernel(CpuKernelContext &ctx, T *output, int64_t num);

 private:
  int64_t cur_step_{0};
  int64_t start_{0};
  std::vector<int64_t> steps_{};
  mindspore::ops::ErrorMode error_mode_{0};

  float flip_probability_{0.};
  int64_t seed_{0};
  int64_t offset_{0};
  int64_t *ele_pos_ptr_{nullptr};
  int64_t ele_pos_num_{0};

  mindspore::ops::FlipMode flip_mode_{0};
  float multiply_factor_{0.};
  int64_t bit_pos_{0};
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_GENERATE_EOD_MASK_V2_H_
