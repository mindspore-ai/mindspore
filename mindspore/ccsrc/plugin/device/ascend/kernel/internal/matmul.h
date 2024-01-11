/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MS_KERNEL_INTERNAL_KERNEL_MATMUL_H_
#define MS_KERNEL_INTERNAL_KERNEL_MATMUL_H_
#include <vector>
#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
namespace mindspore {
namespace kernel {
class MatMul : public InternalKernelMod {
 protected:
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  void SetInOutIdx();
};
}  // namespace kernel
}  // namespace mindspore
#endif
