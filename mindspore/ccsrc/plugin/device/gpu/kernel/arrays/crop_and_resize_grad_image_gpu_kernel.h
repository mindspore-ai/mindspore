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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GRAD_IMAGE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GRAD_IMAGE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include <algorithm>
#include "mindspore/core/ops/crop_and_resize_grad_image.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/crop_and_resize_grad_image_helper.h"
namespace mindspore {
namespace kernel {
constexpr size_t kGrads = 0;
constexpr size_t kBoxes = 1;
constexpr size_t kBoxIndex = 2;
constexpr size_t kImageSize = 3;
class CropAndResizeGradImageGpuKernelMod : public NativeGpuKernelMod {
 public:
  CropAndResizeGradImageGpuKernelMod() { attr_ptr_ = std::make_shared<cukernel::CropAndResizeGradImageAttr>(); }
  ~CropAndResizeGradImageGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_ = nullptr;
  std::shared_ptr<cukernel::CropAndResizeGradImageAttr> attr_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CROP_AND_RESIZE_GRAD_IMAGE_GPU_KERNEL_H_
