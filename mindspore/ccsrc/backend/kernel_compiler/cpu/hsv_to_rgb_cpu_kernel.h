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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_HSV_TO_RGB_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_HSV_TO_RGB_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class HSVToRGBCpuKernel : public CPUKernel {
 public:
  HSVToRGBCpuKernel() = default;
  ~HSVToRGBCpuKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId input_dtype;
  template <typename T1>
  void ConvertOnePixel(T1 h, T1 s, T1 v, T1 *r, T1 *g, T1 *b);
  template <typename T1>
  void ComputeFloat(void *input, void *output, int64_t pixel_num);
  void ComputeHalf(void *input, void *output, int64_t pixel_num);
  std::vector<size_t> shape;
  const size_t kInputNum = 1;
  const size_t kOutputNum = 1;
};
MS_REG_CPU_KERNEL_T(HSVToRGB, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                    HSVToRGBCpuKernel, float16);
MS_REG_CPU_KERNEL_T(HSVToRGB, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    HSVToRGBCpuKernel, float);
MS_REG_CPU_KERNEL_T(HSVToRGB, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    HSVToRGBCpuKernel, double);
}  // namespace kernel
}  // namespace mindspore
#endif
