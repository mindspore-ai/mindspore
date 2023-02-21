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

#ifndef AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_GRAD_KERNELS_H_
#define AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_GRAD_KERNELS_H_

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include "cpu_ops_kernel.h"

namespace aicpu {
struct DeformableOffsetGradDims {
  size_t x_n = 0;
  size_t x_h = 0;
  size_t x_w = 0;
  size_t offset_h = 0;
  size_t offset_w = 0;
  size_t grad_h = 0;
  size_t grad_w = 0;
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t pad_top = 0;
  size_t pad_left = 0;
  size_t stride_h = 0;
  size_t stride_w = 0;
  size_t dilation_h = 0;
  size_t dilation_w = 0;
  size_t deformable_group = 0;
  size_t deformable_group_channel = 0;
};

class DeformableOffsetsGradKernel : public CpuKernel {
 public:
  DeformableOffsetsGradKernel() = default;

  ~DeformableOffsetsGradKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParseKernelParam(const CpuKernelContext &ctx);
  uint32_t CheckInOutNum(size_t inputs_num, size_t outputs_num) const;

  uint32_t SetDims(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoComputeNHWC(const CpuKernelContext &ctx, size_t num_kernels, const DeformableOffsetGradDims &dims,
                         const T *input_x, const T *input_offset, const T *input_grad, T *output_grad_x,
                         T *output_grad_offset) const;
  template <typename T>
  uint32_t DoComputeNCHW(const CpuKernelContext &ctx, size_t num_kernels, const DeformableOffsetGradDims &dims,
                         const T *input_x, const T *input_offset, const T *input_grad, T *output_grad_x,
                         T *output_grad_offset) const;

  template <typename T>
  uint32_t DeformableOffsetsGradTask(const CpuKernelContext &ctx);
  std::string data_format_ = "ND";
  DeformableOffsetGradDims dims_;

  DataType index_type_{DT_FLOAT};
  int64_t index_output_size_ = 1;
  int64_t grad_output_size_ = 1;
  std::vector<int64_t> index_output_shape_;
  std::vector<int64_t> grad_output_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_DEFORMABLE_OFFSETS_GRAD_KERNELS_H_
