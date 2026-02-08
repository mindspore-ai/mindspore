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
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class DynamicReshapeKernelMod : public NativeGpuKernelMod {
 public:
  DynamicReshapeKernelMod() {}
  ~DynamicReshapeKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override {
    MS_LOG(DEBUG) << "Run PostExecute for DynamicReshape, real output shape is " << output_shape_;
    outputs[kIndex0]->SetShapeVector(output_shape_);
    outputs[kIndex0]->set_size(
      LongToSize(std::accumulate(output_shape_.begin(), output_shape_.end(),
                                 UnitSizeInBytes(outputs[kIndex0]->dtype_id()), std::multiplies<int64_t>())));
  }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using LaunchFunc =
    std::function<bool(DynamicReshapeKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
  LaunchFunc kernel_func_;
  ShapeVector output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_
