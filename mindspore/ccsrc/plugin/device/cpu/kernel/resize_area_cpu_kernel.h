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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_AREA_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_AREA_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
struct ResizeAreaCachedInterpolation {
  size_t start;
  size_t end;
  float start_scale;
  float end_minus_one_scale;
  bool needs_bounding = true;
};

class ResizeAreaCPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  ResizeAreaCPUKernelMod() = default;
  ~ResizeAreaCPUKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                    const std::vector<ResizeAreaCachedInterpolation> &x_interps) const;
  template <bool NeedsXBounding, typename T>
  void ComputePatchSum(float scale, const std::vector<const T *> &y_ptrs, const std::vector<float> &y_scales,
                       const ResizeAreaCachedInterpolation &x_interp, float *output_patch_ptr) const;
  float ResizeAreaScaling(size_t in_size, size_t out_size, bool align_corners);
  void SetResizeAreaOutShape(int32_t *out_size) const;

  AnfNodePtr cnode_ptr_;
  TypeId dtype_{kTypeUnknown};
  TypeId size_type_{kTypeUnknown};
  bool align_corners_{false};
  float height_scale_{1.0};
  float width_scale_{1.0};
  int64_t batch_size_{0};
  int64_t channels_{3};
  int64_t in_height_{0};
  int64_t in_width_{0};
  int32_t out_height_{0};
  int32_t out_width_{0};
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_AREA_CPU_KERNEL_H_
