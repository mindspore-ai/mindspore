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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/crop_and_resize_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kImgDimSize = 4;
constexpr size_t kImgBIndex = 0;
constexpr size_t kImgHIndex = 1;
constexpr size_t kImgWIndex = 2;

constexpr size_t kBoxDimSize = 2;
constexpr size_t kCropLengthSize = 2;
constexpr size_t kOutputDimSize = 4;

constexpr size_t kIndexForHeight = 1;
constexpr size_t kIndexForWidth = 2;
constexpr size_t kIndexForChannel = 3;

constexpr size_t kMethodBilinear = 1;
constexpr size_t kMethodNearest = 2;
constexpr size_t kMethodBilinearV2 = 3;
class CropAndResizeGpuKernelMod : public NativeGpuKernelMod {
 public:
  CropAndResizeGpuKernelMod() = default;
  ~CropAndResizeGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using CropAndResizeLaunchFunc = std::function<bool(
    CropAndResizeGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  CropAndResizeLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, CropAndResizeLaunchFunc>> func_list_;

  int method_{0};
  float extrapolation_value_{0};
  int batch_{0};
  int input_height_{0};
  int input_width_{0};
  int final_height_{0};
  int final_width_{0};
  int channel_{0};
  int output_size_{0};
  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CROP_AND_RESIZE_GPU_KERNEL_H_
