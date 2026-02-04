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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_V3_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_V3_GPU_KERNEL_H

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <map>
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_diag_part_v3_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class MatrixDiagPartV3GpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<MatrixDiagPartV3GpuKernelMod> {
 public:
  MatrixDiagPartV3GpuKernelMod() = default;
  ~MatrixDiagPartV3GpuKernelMod() = default;

  using IndexType = int32_t;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  int64_t k_size_{0};
  int64_t num_cols_{0};
  int64_t num_rows_{0};
  int64_t max_diag_len_{0};
  int64_t diag_size_{0};
  bool left_align_super_diag_;
  bool left_align_sub_diag_;
  cudaStream_t cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_V3_GPU_KERNEL_H
