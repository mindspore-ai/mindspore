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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EXTRACT_GLIMPSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EXTRACT_GLIMPSE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <map>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace extract_glimpse_cpu {
class ExtractGlimpseCpuKernelMod : public NativeCpuKernelMod {
 public:
  ExtractGlimpseCpuKernelMod() = default;
  ~ExtractGlimpseCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  std::pair<float, float> GetLocation(const float *ptr, const uint64_t seq,
                                      const std::pair<uint64_t, uint64_t> image_size,
                                      const std::pair<uint64_t, uint64_t> g_size, const bool normalized,
                                      const bool centered);
  void InputsValidCheck(const std::vector<kernel::KernelTensor *> &inputs);
  void Necessity(uint64_t un, bool u_n, float *y_d, uint64_t p_y, std::string no);
  using ExtractGlimpseFunc =
    std::function<bool(ExtractGlimpseCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, ExtractGlimpseFunc>> func_list_;
  TypeId input_shape_type_{kNumberTypeFloat32};
  TypeId size_shape_type_{kNumberTypeInt32};
  TypeId offsets_shape_type_{kNumberTypeFloat32};
  TypeId output_shape_type_{kNumberTypeFloat32};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> size_shape_;
  std::vector<int64_t> offsets_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input_dtype_;
  bool centered_;
  bool normalized_;
  bool uniform_noise_;
  string noise_;
};
}  // namespace extract_glimpse_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EXTRACT_GLIMPSE_CPU_KERNEL_H_
