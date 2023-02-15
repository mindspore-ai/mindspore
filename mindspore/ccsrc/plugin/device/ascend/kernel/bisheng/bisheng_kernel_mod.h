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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H

#include <algorithm>
#include <utility>
#include <vector>
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/bisheng/bisheng_op_info.h"

namespace mindspore {
namespace kernel {
class BiShengKernelMod : public KernelMod {
 public:
  BiShengKernelMod() = default;
  ~BiShengKernelMod() override = default;
};

#define KernelFunc(Clazz)                                                                                         \
 public:                                                                                                          \
  using Func =                                                                                                    \
    std::function<bool(Clazz *, const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, \
                       const std::vector<kernel::AddressPtr> &, void *stream)>;                                   \
  using FuncList = std::vector<std::pair<KernelAttr, Clazz::Func>>;                                               \
  std::vector<KernelAttr> GetOpSupport() override {                                                               \
    std::vector<KernelAttr> support_list;                                                                         \
    (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),                  \
                         [](const std::pair<KernelAttr, Clazz::Func> &pair) { return pair.first; });              \
    return support_list;                                                                                          \
  }                                                                                                               \
                                                                                                                  \
 private:                                                                                                         \
  friend class BishengOpInfoRegister<Clazz>;                                                                      \
  inline static FuncList func_list_ = {};                                                                         \
  static const BishengOpInfoRegister<Clazz> reg_;                                                                 \
  Func kernel_func_;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H
