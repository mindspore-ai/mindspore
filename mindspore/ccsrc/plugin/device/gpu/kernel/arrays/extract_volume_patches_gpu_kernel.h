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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_VOLUME_PATCHES_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_VOLUME_PATCHES_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>

#include "mindspore/core/ops/extract_volume_patches.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_volume_patches_impl.cuh"

namespace mindspore {
namespace kernel {
class ExtractVolumePatchesGpuKernelMod : public NativeGpuKernelMod,
                                         public MatchKernelHelper<ExtractVolumePatchesGpuKernelMod> {
 public:
  ExtractVolumePatchesGpuKernelMod() = default;
  ~ExtractVolumePatchesGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  void *cuda_stream_{nullptr};
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> strides_;
  std::string padding_;
  int64_t ksize_d_{1};
  int64_t ksize_h_{1};
  int64_t ksize_w_{1};
  int64_t stride_d_{1};
  int64_t stride_h_{1};
  int64_t stride_w_{1};
  size_t output_size_{1};
  int64_t input_channel_{1};
  int64_t input_depth_{1};
  int64_t input_height_{1};
  int64_t input_width_{1};
  int64_t output_depth_{1};
  int64_t output_height_{1};
  int64_t output_width_{1};

  int64_t d_stride_{1};
  int64_t h_stride_{1};
  int64_t w_stride_{1};
  int64_t patch_stride_{1};
  int64_t other_stride_{1};
  int64_t chan_input_stride_{1};
  int64_t dep_input_stride_{1};
  int64_t row_input_stride_{1};

  int64_t patch_input_stride_{1};
  bool need_batch_{1};
  int64_t pad_head_{0};
  int64_t pad_top_{0};
  int64_t pad_left_{0};
  bool is_null_input_{false};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_VOLUME_PATCHES_GPU_KERNEL_H_
