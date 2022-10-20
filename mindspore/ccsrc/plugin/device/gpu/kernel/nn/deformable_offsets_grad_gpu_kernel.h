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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DEFORMABLE_OFFSETS_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DEFORMABLE_OFFSETS_GRAD_KERNEL_H_

#include <string>
#include <vector>
#include <map>
#include <utility>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/nn/deformable_offsets_grad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/deformable_offsets_grad_impl.cuh"

namespace mindspore {
namespace kernel {
class DeformableOffsetsGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  DeformableOffsetsGradGpuKernelMod() = default;
  ~DeformableOffsetsGradGpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  struct DeformableOffsetsGradDims {
    uint x_n;
    uint x_h;
    uint x_w;
    uint offset_h;
    uint offset_w;
    uint grad_h;
    uint grad_w;
    uint kernel_h;
    uint kernel_w;
    uint pad_top;
    uint pad_left;
    uint stride_h;
    uint stride_w;
    uint dilation_h;
    uint dilation_w;
    uint deformable_group;
    uint deformable_group_channel;
  };
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  using KernelFunc = std::function<bool(DeformableOffsetsGradGpuKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &)>;

  void SetDims(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
               const std::vector<KernelTensorPtr> &outputs);

  cudaStream_t cuda_stream_{nullptr};
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;
  std::string data_format_ = kOpFormat_NCHW;
  DeformableOffsetsGradDims dims_;
  size_t grad_x_size_{0};
  size_t grad_offset_size_{0};
  size_t type_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DEFORMABLE_OFFSETS_GRAD_KERNEL_H_
