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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2NORMALIZE_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2NORMALIZE_GRAD_CPU_KERNEL_H_

#include <vector>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class L2NormalizeGradCPUKernel : public CPUKernel {
 public:
  L2NormalizeGradCPUKernel() = default;
  ~L2NormalizeGradCPUKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void InitKernel(const CNodePtr &kernel_node) override;

 private:
  void CheckInputShape(const std::vector<size_t> &output_shape);
  std::vector<size_t> OneDimIndexToHighDimIndex(size_t one_dim_index);
  void HighDimIndexToOneDimIndex(size_t *one_dim_index, const std::vector<size_t> &high_dim_index);
  std::vector<T> GetVector(const std::vector<size_t> &high_dim_index, const T *x);
  void GetSumOfProduct(const std::vector<T> &x_vector, const std::vector<T> &y_vector, T *ss);
  void GetOutput(const std::vector<T> &input_x_vector, const std::vector<T> &y_vector,
                 const std::vector<T> &dout_vector, const std::vector<size_t> &high_dim_index, T *output);
  std::vector<std::vector<size_t>> input_shape_list_;
  std::vector<size_t> dim_elem_num_list_;
  int axis_{0};
  T epsilon_{0};
};

MS_REG_CPU_KERNEL_T(L2NormalizeGrad,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    L2NormalizeGradCPUKernel, float);

MS_REG_CPU_KERNEL_T(L2NormalizeGrad,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddOutputAttr(kNumberTypeFloat16),
                    L2NormalizeGradCPUKernel, float16);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2NORMALIZE_GRAD_CPU_KERNEL_H_
