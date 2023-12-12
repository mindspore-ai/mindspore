/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "kernel/kernel.h"
#include "kernel/pyboost/pyboost_kernel_extra_func.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT PyboostCPUKernelExtraFunc : public PyboostKernelExtraFunc {
 public:
  void SetThreadPool(const kernel::KernelModPtr &kernel) override;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
