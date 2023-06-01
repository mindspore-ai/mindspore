/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_UPSAMPLE_TRILINEAR3D_H
#define AICPU_KERNELS_NORMALIZED_UPSAMPLE_TRILINEAR3D_H
#include <cmath>
#include <limits>

#include "cpu_ops_kernel.h"
namespace aicpu {
class UpsampleTrilinear3dCpuKernel : public CpuKernel {
 public:
  ~UpsampleTrilinear3dCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t UpsampleTrilinear3dParamCheck(CpuKernelContext &ctx);

  template <typename T, typename S>
  uint32_t UpsampleTrilinear3dCompute(const CpuKernelContext &ctx);

  template <typename T, typename S>
  void InnerCompute(int64_t n, const T *x_ptr, T *y_ptr);

  int64_t input_depth;
  int64_t input_height;
  int64_t input_width;

  int64_t output_depth;
  int64_t output_height;
  int64_t output_width;

  int64_t output_slice_size;
  int64_t input_slice_size;
  bool align_corners = false;
  std::vector<float> scales{0., 0., 0.};
  std::vector<int64_t> none_list;
};
}  // namespace aicpu
#endif
