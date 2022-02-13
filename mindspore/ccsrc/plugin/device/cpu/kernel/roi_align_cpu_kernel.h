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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_CPU_KERNEL_H_
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr int ROIS_COLS = 5;
constexpr size_t X_DIMS = 4;
constexpr int CHANNEL = 1;
constexpr int HEIGHT = 2;
constexpr int WIDTH = 3;
template <typename T>
class ROIAlignCpuKernelMod : public NativeCpuKernelMod {
 public:
  ROIAlignCpuKernelMod() = default;
  ~ROIAlignCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  void bilinear_interpolate(const int height, const int width, T y, T x, int *x_low, int *y_low, int *x_high,
                            int *y_high, T *w1, T *w2, T *w3, T *w4);

  void bin_box(int thread_idx, const T *roi_boxes, int roi_cols, const T spatial_scale, const int sample_num,
               int roi_end_mode, const int channels, const int height, const int width, const int pooled_height,
               const int pooled_width, int *offset, int *n, int *c, int *ph, int *pw, int *roi_bin_grid_h,
               int *roi_bin_grid_w, T *bin_size_h, T *bin_size_w, T *roi_start_h, T *roi_start_w);

  int pooled_height_{0};
  int pooled_width_{0};
  T spatial_scale_{0.0};
  int sample_num_{0};
  int roi_end_mode_{0};

  int roi_rows_{0};
  int roi_cols_{0};
  int channels_{0};
  int height_{0};
  int width_{0};
};

MS_REG_CPU_KERNEL_T(
  ROIAlign,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ROIAlignCpuKernelMod, float);

MS_REG_CPU_KERNEL_T(
  ROIAlign,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ROIAlignCpuKernelMod, float16);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_CPU_KERNEL_H_
