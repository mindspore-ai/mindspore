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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseAddGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseAddGpuKernelMod() = default;
  ~SparseAddGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  void SyncData() override;
  std::vector<KernelAttr> GetOpSupport() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  void ResetResource() noexcept;
  void CalWorkSpace();
  template <typename T, typename S, typename K>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using SparseAddLaunchFunc =
    std::function<bool(SparseAddGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseAddLaunchFunc>> func_list_;
  SparseAddLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
  size_t a_indices_size_ = 0;
  size_t a_values_size_ = 0;
  size_t dense_shape_size_ = 0;
  size_t b_indices_size_ = 0;
  size_t b_values_size_ = 0;
  size_t indices_size_ = 0;       // That is, sizeof(T).
  size_t values_size_ = 0;        // That is, sizeof(S)
  size_t threshold_size_ = 0;     // That is, sizeof(K)
  int64_t real_output_size_ = 0;  // Dynamic shape related.
  std::vector<size_t> a_indices_shape_{};
  std::vector<size_t> a_values_shape_{};
  std::vector<size_t> dense_shape_{};
  std::vector<size_t> b_indices_shape_{};
  std::vector<size_t> b_values_shape_{};
  std::vector<KernelTensorPtr> outputs_{};
  size_t rank_ = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_ADD_GPU_KERNEL_H_
