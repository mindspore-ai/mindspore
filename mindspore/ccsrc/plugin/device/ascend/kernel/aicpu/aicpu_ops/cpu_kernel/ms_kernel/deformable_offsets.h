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

#ifndef AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_KERNELS_H_
#define AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_KERNELS_H_

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "cpu_ops_kernel.h"

namespace aicpu {
class DeformableOffsetsKernel : public CpuKernel {
 public:
  DeformableOffsetsKernel() = default;

  ~DeformableOffsetsKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  void ResetResource() noexcept;

  uint32_t ParseKernelParam(const CpuKernelContext &ctx);

  uint32_t ParseAttrs(const CpuKernelContext &ctx);
  uint32_t SetDims(const CpuKernelContext &ctx);

  uint32_t GenPositionGrid(const CpuKernelContext &ctx, int64_t *position_grid);

  template <typename T>
  uint32_t DoCompute(const CpuKernelContext &ctx, const int64_t *position_grid);

  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> dilations_;
  std::vector<size_t> workspace_size_list_;
  int64_t deformable_groups_{1};
  bool modulated_{true};

  size_t n_axis_{0};
  size_t c_axis_{1};
  size_t h_axis_{2};
  size_t w_axis_{3};
  int64_t n_{0};
  int64_t c_{0};
  int64_t input_h_{0};
  int64_t input_w_{0};
  int64_t output_h_{0};
  int64_t output_w_{0};
  int64_t position_grid_size_{0};
  DataType index_type_{DT_FLOAT};
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_KERNELS_H_
