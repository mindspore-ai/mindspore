/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/nn/maxpool_with_argmax_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(MaxPoolGradWithArgmax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      MaxPoolWithArgmaxGradGpuKernel, float, int)
MS_REG_GPU_KERNEL_TWO(MaxPoolGradWithArgmax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      MaxPoolWithArgmaxGradGpuKernel, half, int)
}  // namespace kernel
}  // namespace mindspore
