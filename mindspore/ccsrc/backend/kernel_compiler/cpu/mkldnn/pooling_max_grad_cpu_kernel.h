/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_MAX_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_MAX_GRAD_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MaxPoolingGradCPUKernel : public MKLCPUKernel {
 public:
  MaxPoolingGradCPUKernel() = default;
  ~MaxPoolingGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void RowPoolingGrad(const float *input, float *output, float diff, const std::vector<std::pair<size_t, size_t>> &box,
                      std::vector<std::pair<size_t, float>> *row_max_pair);
  void ChannelPoolingGrad(const float *input, const float *diff, float *output);
  std::vector<int> stride_;
  std::vector<size_t> kernel_size_;
  std::vector<int> padding_l_;
  std::vector<size_t> src_shape_;
  std::vector<size_t> dst_shape_;
};

MS_REG_CPU_KERNEL(MaxPoolGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  MaxPoolingGradCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_MAX_GRAD_CPU_KERNEL_H_
