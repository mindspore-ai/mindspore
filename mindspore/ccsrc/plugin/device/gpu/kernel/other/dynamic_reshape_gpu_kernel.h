/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class DynamicReshapeKernelMod : public NativeGpuKernelMod {
 public:
  DynamicReshapeKernelMod() {}
  ~DynamicReshapeKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    is_need_retrieve_output_shape_ = true;
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

 protected:
  std::vector<KernelTensorPtr> outputs_{};
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }
  void SyncData() override {
    MS_LOG(DEBUG) << "Run PostExecute for DynamicReshape, real output shape is " << output_shape_;
    outputs_[kIndex0]->SetShapeVector(output_shape_);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using LaunchFunc = std::function<bool(DynamicReshapeKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
  LaunchFunc kernel_func_;
  ShapeVector output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_
