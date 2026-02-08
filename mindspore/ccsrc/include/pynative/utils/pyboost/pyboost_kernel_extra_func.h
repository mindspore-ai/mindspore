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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_KERNEL_EXTRA_FUNC_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_KERNEL_EXTRA_FUNC_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class PYBOOST_API PyboostKernelExtraFunc {
 public:
  virtual ~PyboostKernelExtraFunc() = default;
  virtual void SetThreadPool(const kernel::KernelModPtr &kernel) {}
  virtual bool IsPyBoostCustomRegistered(const std::string &op_name) = 0;
  virtual bool IsKernelModRegistered(const std::string &op_name) = 0;
  virtual bool IsEnableProfiler() = 0;
  virtual void LaunchKernelWithProfiler(const std::string &op_name, const device::DeviceContext *device_context,
                                        const std::vector<BaseShapePtr> &base_shape,
                                        const std::function<void()> &func) = 0;
};

using PyboostKernelExtraFuncPtr = std::shared_ptr<PyboostKernelExtraFunc>;
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_KERNEL_EXTRA_FUNC_H_
