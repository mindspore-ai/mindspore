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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_MATRIX_TRANSPOSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_MATRIX_TRANSPOSE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <complex>
#include "kernel/gpu/cuda_impl/cuda_ops/csr_sparse_matrix_to_sparse_tensor_gpu_kernel.cuh"
#include "plugin/gpu/res_manager/symbol_interface/cuda_driver.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class SparseMatrixTransposeGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseMatrixTransposeGpuKernelMod() {
    ResetResource();
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle();
  }
  ~SparseMatrixTransposeGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {};
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S>
  size_t GetBufferSize(size_t num_batches, int rows, int cols, const std::vector<S> &host_batch_pointers, S *x_row_ptrs,
                       S *x_col_inds, S *y_row_ptrs, S *y_col_inds);
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using SparseMatrixTransposeLaunchFunc =
    std::function<bool(SparseMatrixTransposeGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseMatrixTransposeLaunchFunc>> func_list_;
  SparseMatrixTransposeLaunchFunc kernel_func_;

  cudaStream_t cuda_stream_;
  cusparseHandle_t handle_{nullptr};

  // shapes
  std::vector<int64_t> input_dense_shape_shapes_;
  std::vector<int64_t> input_batch_pointers_shapes_;
  std::vector<int64_t> input_row_pointers_shapes_;
  std::vector<int64_t> input_col_indices_shapes_;
  std::vector<int64_t> input_values_shapes_;

  std::vector<int64_t> output_dense_shape_shapes_;
  std::vector<int64_t> output_batch_pointers_shapes_;
  std::vector<int64_t> output_row_pointers_shapes_;
  std::vector<int64_t> output_col_indices_shapes_;
  std::vector<int64_t> output_values_shapes_;

  // others
  bool batched{false};
  int rank_;
  TypeId ms_type_;
  bool conjugate{false};
  bool is_null_input_{false};
  bool is_empty_matrix{false};
  size_t buffer_size{0};
  size_t output_values_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_MATRIX_TRANSPOSE_GPU_KERNEL_H_
