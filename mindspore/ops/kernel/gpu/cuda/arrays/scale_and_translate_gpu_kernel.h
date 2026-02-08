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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCALE_AND_TRANSLATE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCALE_AND_TRANSLATE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/scale_and_translate.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_class/scale_and_translate_helper.h"

namespace mindspore {
namespace kernel {
constexpr size_t kImages = 0;
constexpr size_t kSize = 1;
constexpr size_t kScale = 2;
constexpr size_t kTranslation = 3;

class ScaleAndTranslateGpuKernelMod : public NativeGpuKernelMod {
 public:
  ScaleAndTranslateGpuKernelMod() { attr_ptr_ = std::make_shared<cukernel::ScaleAndTranslateAttr>(); }
  ~ScaleAndTranslateGpuKernelMod() override = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_{nullptr};
  std::shared_ptr<cukernel::ScaleAndTranslateAttr> attr_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCALE_AND_TRANSLATE_GPU_KERNEL_H_
