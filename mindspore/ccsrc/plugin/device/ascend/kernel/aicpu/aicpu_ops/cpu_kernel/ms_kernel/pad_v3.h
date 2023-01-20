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
#ifndef AICPU_KERNELS_NORMALIZED_PAD_V3_H_
#define AICPU_KERNELS_NORMALIZED_PAD_V3_H_

#include <memory>
#include <utility>
#include <vector>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/bcast.h"

namespace aicpu {
class PadV3CpuKernel : public CpuKernel {
 public:
  PadV3CpuKernel() = default;
  ~PadV3CpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  std::vector<int64_t> paddings;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::string mode;
  bool paddings_contiguous;
  int64_t input_dims{0};
  int64_t paddings_num{0};
  int64_t parallelSliceNum{1};

  uint32_t CheckAndInitParams(CpuKernelContext &ctx);

  template <typename T>
  uint32_t GetPaddingsAndSetOuputShape(CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t EdgeModeCompute(CpuKernelContext &ctx, int64_t p);

  template <typename T>
  uint32_t EdgeCompute3D(T *input, T *output, int64_t p);

  template <typename T>
  uint32_t EdgeCompute2D(T *input, T *output, int64_t p);

  template <typename T>
  uint32_t EdgeCompute1D(T *input, T *output, int64_t p);

  int64_t EdgeIndexCaculate(int64_t pad_value, int64_t now, int64_t input_value, int64_t o_start, int64_t i_start);

  template <typename T>
  uint32_t ReflectModeCompute(CpuKernelContext &ctx, int64_t p);

  template <typename T>
  uint32_t ReflectCompute3D(T *input, T *output, int64_t p);

  template <typename T>
  uint32_t ReflectCompute2D(T *input, T *output, int64_t p);

  template <typename T>
  uint32_t ReflectCompute1D(T *input, T *output, int64_t p);

  int64_t ReflectIndexCaculate(int64_t pad_value, int64_t now, int64_t input_value, int64_t o_start, int64_t i_start);

  template <typename T>
  uint32_t ConstantModeCompute(CpuKernelContext &ctx, T constant_values);
};
}  // namespace aicpu
#endif
