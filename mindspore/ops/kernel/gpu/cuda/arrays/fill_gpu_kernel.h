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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_FILL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_FILL_GPU_KERNEL_H_
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
namespace mindspore {
namespace kernel {
class FillGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<FillGpuKernelMod> {
 public:
  FillGpuKernelMod() = default;
  ~FillGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kShapeIndex_}; }

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  template <typename T>
  T GetInputDataFromDevice(const std::vector<KernelTensor *> &inputs, size_t idx, cudaStream_t cuda_stream);

  TypeId x_type_id_;
  size_t input_elements_{0};
  static constexpr size_t kShapeIndex_{0};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_FILL_GPU_KERNEL_H_
