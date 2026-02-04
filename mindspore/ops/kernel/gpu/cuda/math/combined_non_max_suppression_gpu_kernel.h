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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_COMBINED_NON_MAX_SUPPRESSION_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_COMBINED_NON_MAX_SUPPRESSION_GPU_KERNEL_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "mindspore/ops/infer/combined_non_max_suppression.h"
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class CombinedNonMaxSuppressionGpuKernelMod : public NativeGpuKernelMod {
 public:
  CombinedNonMaxSuppressionGpuKernelMod() { ResetResource(); }
  ~CombinedNonMaxSuppressionGpuKernelMod() override = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using CombinedNonMaxSuppressionLaunchFunc =
    std::function<bool(CombinedNonMaxSuppressionGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, CombinedNonMaxSuppressionLaunchFunc>> func_list_;
  CombinedNonMaxSuppressionLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
  void InitSizeLists();
  size_t T{0};
  int batch_size_{0};
  int num_boxes_{0};
  int q_{0};
  int num_classes_{0};
  int per_detections_{0};
  bool pad_per_class_{false};
  bool clip_boxes_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_COMBINED_NON_MAX_SUPPRESSION_GPU_KERNEL_H_
