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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_Split_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_Split_GPU_KERNEL_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "mindspore/core/ops/sparse_split.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_split_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseSplitGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseSplitGpuKernelMod() = default;
  ~SparseSplitGpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using SparseSplitLaunchFunc =
    std::function<bool(SparseSplitGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseSplitLaunchFunc>> func_list_;
  SparseSplitLaunchFunc kernel_func_;
  std::vector<KernelTensorPtr> outputs_{};
  cudaStream_t cuda_stream;
  int64_t real_output_size = 0;
  std::vector<int64_t> h_block;
  int64_t h_split_dim;
  std::vector<int> h_blocks;
  size_t input_nnz_{0};
  size_t num_dim_{0};
  size_t out_size_{0};
  size_t num_split;
  TypeId input_dtype_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_SPARSE_SEGMENT_MEAN_GPU_KERNEL_H_
