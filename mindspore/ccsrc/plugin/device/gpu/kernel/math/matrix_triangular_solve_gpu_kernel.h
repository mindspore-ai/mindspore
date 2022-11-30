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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixTriangularSolveGpuKernelMod : public NativeGpuKernelMod,
                                          public MatchKernelHelper<MatrixTriangularSolveGpuKernelMod> {
 public:
  MatrixTriangularSolveGpuKernelMod() = default;
  ~MatrixTriangularSolveGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  bool is_null_input_{false};
  bool lower_{true};
  bool adjoint_{false};
  bool is_bcast_required_{false};
  uint32_t device_id_{0};
  size_t unit_size_;
  int64_t m_;
  int64_t n_;
  int64_t a_batch_num_;
  int64_t b_batch_num_;
  int64_t output_batch_num_;
  std::vector<int64_t> a_broadcast_indices_;
  std::vector<int64_t> b_broadcast_indices_;
  cublasSideMode_t side_ = CUBLAS_SIDE_RIGHT;
  cublasDiagType_t diag_ = CUBLAS_DIAG_NON_UNIT;
  cublasFillMode_t uplo_;
  cublasOperation_t trans_;
  cublasHandle_t blas_handle_{nullptr};
  cudaStream_t cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_TRIANGULAR_SOLVE_GPU_KERNEL_H_
