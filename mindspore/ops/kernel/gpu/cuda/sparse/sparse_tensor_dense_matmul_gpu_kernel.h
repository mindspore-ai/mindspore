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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_SPARSE_TENSOR_DENSE_MATMUL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_SPARSE_TENSOR_DENSE_MATMUL_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/ops/infer/sparse_tensor_dense_mat_mul.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_class/sparse_tensor_dense_matmul_helper.h"

namespace mindspore {
namespace kernel {
class SparseTensorDenseMatmulGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseTensorDenseMatmulGpuKernelMod() { attr_ptr_ = std::make_shared<cukernel::SparseTensorDenseMatmulAttr>(); }
  ~SparseTensorDenseMatmulGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_{nullptr};
  std::shared_ptr<cukernel::SparseTensorDenseMatmulAttr> attr_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_SPARSE_TENSOR_DENSE_MATMUL_GPU_KERNEL_H_
