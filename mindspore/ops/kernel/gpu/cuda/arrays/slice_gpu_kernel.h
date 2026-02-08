/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_class/slice_helper.h"

namespace mindspore {
namespace kernel {
class SliceGpuKernelMod : public NativeGpuKernelMod {
 public:
  SliceGpuKernelMod() {
    kernel_name_ = "Slice";
    attr_ptr_ = std::make_shared<cukernel::SliceAttr>();
  }
  ~SliceGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;
  // std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kBeginIndex_, kSizeIndex_}; }

 private:
  void CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void ProccessAttr(const std::vector<KernelTensor *> &inputs);
  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_{nullptr};
  std::shared_ptr<cukernel::SliceAttr> attr_ptr_{nullptr};
  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  static constexpr size_t kBeginIndex_{1};
  static constexpr size_t kSizeIndex_{2};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_
