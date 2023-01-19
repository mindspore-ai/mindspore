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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_DENSE_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_DENSE_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include <map>
#include <utility>
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dense_to_csr_sparse_matrix_gpu_kernel.cuh"

namespace mindspore {
namespace kernel {
class DenseToCSRSparseMatrixKernelMod : public NativeGpuKernelMod {
 public:
  DenseToCSRSparseMatrixKernelMod() { handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle(); }
  ~DenseToCSRSparseMatrixKernelMod() {}

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_name = base_operator->GetPrim()->name();
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  cusparseHandle_t handle_{nullptr};
  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> dense_shape_shapes_;
  std::vector<size_t> batch_pointers_shapes_;
  std::vector<size_t> row_pointers_shapes_;
  std::vector<size_t> col_indices_shapes_;
  std::vector<size_t> value_shapes_;
  std::vector<size_t> dims_;
  size_t dim_indices_last_;

  bool is_batch_csr_{false};
  int nnz_;
  int m_;
  int rank_;

  void Reshape() {
    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); i++) {
      dim_of_indices *= indices_shapes_[i];
    }

    size_t dim_after_indices = 1;
    size_t dim_indices_last = indices_shapes_[indices_shapes_.size() - IntToSize(1)];
    for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
      dim_after_indices *= input_shapes_[i];
    }
    dims_ = {dim_of_indices, dim_after_indices, dim_indices_last};
  }

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using LaunchFunc = std::function<bool(DenseToCSRSparseMatrixKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
  LaunchFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
