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
#include "mindspore/core/ops/combined_non_max_suppression.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CombinedNonMaxSuppressionGpuKernelMod : public NativeGpuKernelMod {
 public:
  CombinedNonMaxSuppressionGpuKernelMod() { ResetResource(); }
  ~CombinedNonMaxSuppressionGpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  void SyncOutputShape() override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using CombinedNonMaxSuppressionLaunchFunc =
    std::function<bool(CombinedNonMaxSuppressionGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, CombinedNonMaxSuppressionLaunchFunc>> func_list_;
  CombinedNonMaxSuppressionLaunchFunc kernel_func_;
  cudaStream_t cuda_stream_;
  void InitSizeLists();
  size_t T;
  int batch_size_;
  int num_boxes_;
  int q_;
  int num_classes_;
  int per_detections_;
  bool pad_per_class_;
  bool clip_boxes_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_COMBINED_NON_MAX_SUPPRESSION_GPU_KERNEL_H_
