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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MAP_TENSOR_MAP_TENSOR_GET_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MAP_TENSOR_MAP_TENSOR_GET_GRAD_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/map_tensor/map_tensor_gpu_kernel.h"
#include "plugin/device/gpu/hal/device/gpu_hash_table.h"

namespace mindspore {
namespace kernel {
using device::gpu::GPUHashTable;
constexpr size_t kMapTensorGetGradInputNum = 3;
constexpr size_t kMapTensorGetGradOutputNum = 1;

class MapTensorGetGradGpuKernelMod : public MapTensorGpuKernelMod {
 public:
  MapTensorGetGradGpuKernelMod() = default;
  ~MapTensorGetGradGpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetGradInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetGradOutputNum, kernel_name_);
    return kernel_launch_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  template <typename KeyType>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  bool InitSize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs);

  using MapTensorGetGradLaunchFunc =
    std::function<bool(MapTensorGetGradGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, MapTensorGetGradLaunchFunc>> map_tensor_get_grad_func_list_;
  MapTensorGetGradLaunchFunc kernel_launch_func_;

  std::vector<KernelTensorPtr> outputs_ = {};
  int64_t key_size_{1};
  ShapeVector value_dims_ = {};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MAP_TENSOR_MAP_TENSOR_GET_GRAD_GPU_KERNEL_H_
