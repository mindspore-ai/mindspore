/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/oplib/oplib.h"
#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::device::cpu::KernelAttr;
using NativeCpuKernelModCreator = std::function<std::shared_ptr<NativeCpuKernelMod>()>;

class NativeCpuKernelModFactory {
 public:
  static NativeCpuKernelModFactory &GetInstance();
  void Register(const std::string &kernel_name, const KernelAttr &kernel_attr,
                NativeCpuKernelModCreator &&kernel_creator);
  std::shared_ptr<NativeCpuKernelMod> Create(const std::string &kernel_name, const CNodePtr &apply_kernel);
  void SetKernelAttrs(const std::shared_ptr<kernel::OpInfo> op_info, std::vector<KernelAttr> *kernel_attrs);
  void UpdateKernelAttrs(const std::string &kernel_name, const std::vector<KernelAttr> &kernel_attrs);
  std::vector<KernelAttr> GetSupportedKernelAttrList(const std::string &kernel_name);
  bool SearchRegisteredOp(const std::string &kernel_name) const;

 private:
  NativeCpuKernelModFactory() = default;
  ~NativeCpuKernelModFactory() = default;
  DISABLE_COPY_AND_ASSIGN(NativeCpuKernelModFactory)
  std::pair<bool, size_t> CPUKernelAttrCheck(const std::string &kernel_name, const KernelBuildInfo &kernel_info);
  bool CPUKernelSingleAttrCheck(const KernelAttr &kernel_attr, const KernelBuildInfo &kernel_info) const;

  // Set output and input ref map to kernel info which will be used by graph compiler.
  void SetRefMapToKernelInfo(const std::string &kernel_name, size_t index, device::KernelInfo *kernel_info);

  std::map<std::string, std::vector<std::pair<KernelAttr, NativeCpuKernelModCreator>>> name_to_attr_creator_;
};

class NativeCpuKernelRegistrar {
 public:
  NativeCpuKernelRegistrar(const std::string &kernel_name, const KernelAttr &kernel_attr,
                           NativeCpuKernelModCreator &&kernel_creator) {
    NativeCpuKernelModFactory::GetInstance().Register(kernel_name, kernel_attr, std::move(kernel_creator));
  }
  ~NativeCpuKernelRegistrar() = default;
};

#define MS_REG_CPU_KERNEL(OPNAME, ATTR, OPCLASS) MS_REG_CPU_KERNEL_(__COUNTER__, OPNAME, ATTR, OPCLASS)
#define MS_REG_CPU_KERNEL_(COUNT, OPNAME, ATTR, OPCLASS) _MS_REG_CPU_KERNEL_(COUNT, OPNAME, ATTR, OPCLASS)
#define _MS_REG_CPU_KERNEL_(COUNT, OPNAME, ATTR, OPCLASS)                                                    \
  static_assert(std::is_base_of<NativeCpuKernelMod, OPCLASS>::value, " must be base of NativeCpuKernelMod"); \
  static const NativeCpuKernelRegistrar g_cpu_kernel_##COUNT##_reg(#OPNAME, ATTR,                            \
                                                                   []() { return std::make_shared<OPCLASS>(); });

#define MS_REG_CPU_KERNEL_T(OPNAME, ATTR, OPCLASS, T) MS_REG_CPU_KERNEL_T_(__COUNTER__, OPNAME, ATTR, OPCLASS, T)
#define MS_REG_CPU_KERNEL_T_(COUNT, OPNAME, ATTR, OPCLASS, T) _MS_REG_CPU_KERNEL_T_(COUNT, OPNAME, ATTR, OPCLASS, T)
#define _MS_REG_CPU_KERNEL_T_(COUNT, OPNAME, ATTR, OPCLASS, T)                                                  \
  static_assert(std::is_base_of<NativeCpuKernelMod, OPCLASS<T>>::value, " must be base of NativeCpuKernelMod"); \
  static const NativeCpuKernelRegistrar g_cpu_kernel_##COUNT##_##OPNAME##_##T##_reg(                            \
    #OPNAME, ATTR, []() { return std::make_shared<OPCLASS<T>>(); });

#define MS_REG_CPU_KERNEL_T_S(OPNAME, ATTR, OPCLASS, T, S) \
  MS_REG_CPU_KERNEL_T_S_(__COUNTER__, OPNAME, ATTR, OPCLASS, T, S)
#define MS_REG_CPU_KERNEL_T_S_(COUNT, OPNAME, ATTR, OPCLASS, T, S) \
  _MS_REG_CPU_KERNEL_T_S_(COUNT, OPNAME, ATTR, OPCLASS, T, S)
#define _MS_REG_CPU_KERNEL_T_S_(COUNT, OPNAME, ATTR, OPCLASS, T, S)                                                \
  static_assert(std::is_base_of<NativeCpuKernelMod, OPCLASS<T, S>>::value, " must be base of NativeCpuKernelMod"); \
  static const NativeCpuKernelRegistrar g_cpu_kernel_##COUNT##_##OPNAME##_##T##_##S##_reg(                         \
    #OPNAME, ATTR, []() { return std::make_shared<OPCLASS<T, S>>(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_FACTORY_H_
