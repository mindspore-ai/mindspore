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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_
#include <vector>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class CropAndResizeCPUKernel : public CPUKernel {
 public:
  CropAndResizeCPUKernel() = default;
  ~CropAndResizeCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  int method_;
  float extrapolation_value_;
  int input_crop_size_;
  int output_size_;
  int input_height_;
  int input_width_;
  int final_height_;
  int final_width_;
  int channel_;
};

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, float16);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, float16);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, float);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, float);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, double);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, double);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt8)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int8_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt8)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int8_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int16_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int16_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt8)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int8_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int32_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt8)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, uint8_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt8)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, uint8_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, uint16_t);

MS_REG_CPU_KERNEL_T(CropAndResize,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt16)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    CropAndResizeCPUKernel, uint16_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_
