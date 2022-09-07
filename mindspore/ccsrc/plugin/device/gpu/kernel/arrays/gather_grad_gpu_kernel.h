/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
#define MINDSPORE_GATHER_GRAD_GPU_KERNEL_H

#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include <vector>
#include "ops/grad/gather_d_grad.h"
#include "ops/grad/gather_d_grad_v2.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather_grad.cuh"

namespace mindspore {
namespace kernel {
class GatherGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit GatherGradGpuKernelMod(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  ~GatherGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs, void *stream_ptr);

 private:
  using GatherGradOpFunc = std::function<bool(GatherGradGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &, void *)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, GatherGradGpuKernelMod::GatherGradOpFunc>>>
    kernel_attr_map_;
  GatherGradOpFunc kernel_func_;

  ShapeVector index_shapes_;
  ShapeVector grad_shapes_;
  ShapeVector output_shapes_;

  std::string kernel_name_;
  size_t dims_[4] = {};
  int axis_{0};
  void *cuda_stream_{nullptr};
  size_t index_idx_{0};
  size_t grad_idx_{0};
  size_t idx_type_size_{0};
  size_t grad_type_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
