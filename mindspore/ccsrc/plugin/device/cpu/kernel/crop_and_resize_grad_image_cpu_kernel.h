/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_IMAGE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_IMAGE_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CropAndResizeGradImageCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  CropAndResizeGradImageCpuKernelMod() = default;
  ~CropAndResizeGradImageCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using CropAndResizeGradImageFunc =
    std::function<bool(CropAndResizeGradImageCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, CropAndResizeGradImageFunc>> func_list_;
  CropAndResizeGradImageFunc kernel_func_;
  template <typename T>
  void GradOfImageCompute(const float *grads, const float *boxes, const int *box_ind, const int *image_size,
                          T *outputDatas, size_t start, size_t end);
  std::vector<int64_t> grads_shape_;
  std::vector<int64_t> image_size_shape_;
  std::vector<int64_t> boxes_shape_;
  std::vector<int64_t> box_ind_shape_;
  std::vector<int64_t> output_shape_;
  std::string attr_method_ = "bilinear";
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_IMAGE_CPU_KERNEL_H_
