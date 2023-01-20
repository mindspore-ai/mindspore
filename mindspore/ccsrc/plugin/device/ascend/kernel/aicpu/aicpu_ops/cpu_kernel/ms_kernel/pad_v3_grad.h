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
#ifndef AICPU_KERNELS_NORMALIZED_PAD_V3_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_PAD_V3_GRAD_H_

#include <vector>

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class PadV3GradCpuKernel : public CpuKernel {
 public:
  PadV3GradCpuKernel() = default;
  ~PadV3GradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  bool padding_contiguous = true;
  std::string mode = "reflect";
  std::vector<int64_t> paddings;
  int64_t output_w;
  int64_t output_h;
  int64_t output_c;
  int64_t input_w;
  int64_t input_h;
  int64_t input_c;
  int64_t i_start_x;
  int64_t i_start_y;
  int64_t i_start_z;
  int64_t o_start_x;
  int64_t o_start_y;
  int64_t o_start_z;
  int64_t pad_l;
  int64_t pad_t;
  int64_t pad_f;
  int64_t parallelSliceNum;
  int64_t num_elem;
  int64_t input_dim;
  uint32_t PadV3GradCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t PadV3GradCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t PadV3GradCompute1D(T *input, T *output, int64_t p);

  template <typename T>
  uint32_t PadV3GradCompute2D(T *input, T *output, int64_t p, int64_t i);

  template <typename T>
  uint32_t PadV3GradCompute3D(T *input, T *output, int64_t p, int64_t z);

  template <typename T>
  uint32_t PadV3GradCompute1(T *input, T *output, int64_t p);

  int64_t IndexCaculate(int64_t pad_value, int64_t now, int64_t output_value, int64_t o_start, int64_t i_start);

  template <typename T>
  uint32_t PadV3ReadPaddingsAndSetOutputShape1(CpuKernelContext &ctx);

  template <typename T>
  uint32_t PadV3ReadPaddingsAndSetOutputShape2(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
