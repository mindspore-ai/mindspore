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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_CPU_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_CPU_KERNEL_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/utils.h"
#include "kernel/cpu/cpu_kernel.h"
#include "device/cpu/kernel_select_cpu.h"

namespace mindspore {
namespace kernel {
using mindspore::device::cpu::KernelAttr;
using CPUKernelCreator = std::function<std::shared_ptr<CPUKernel>()>;
class CPUKernelFactory {
 public:
  static CPUKernelFactory &GetInstance();
  void Register(const std::string &kernel_name, const KernelAttr &kernel_attr, CPUKernelCreator &&kernel_creator);
  std::shared_ptr<CPUKernel> Create(const std::string &kernel_name);
  std::shared_ptr<CPUKernel> Create(const std::string &kernel_name, const CNodePtr &apply_kernel);
  std::vector<KernelAttr> GetSupportedKernelAttrList(const std::string &kernel_name);

 private:
  CPUKernelFactory() = default;
  ~CPUKernelFactory() = default;
  DISABLE_COPY_AND_ASSIGN(CPUKernelFactory)
  std::pair<bool, size_t> CPUKernelAttrCheck(const std::string &kernel_name, const KernelBuildInfo *kernel_info);
  std::map<std::string, std::vector<std::pair<KernelAttr, CPUKernelCreator>>> name_to_attr_creator_;
};

class CPUKernelRegistrar {
 public:
  CPUKernelRegistrar(const std::string &kernel_name, const KernelAttr &kernel_attr, CPUKernelCreator &&kernel_creator) {
    CPUKernelFactory::GetInstance().Register(kernel_name, kernel_attr, std::move(kernel_creator));
  }
  ~CPUKernelRegistrar() = default;
};

#define MS_REG_CPU_KERNEL(OPNAME, ATTR, OPCLASS)                                           \
  static_assert(std::is_base_of<CPUKernel, OPCLASS>::value, " must be base of CPUKernel"); \
  static const CPUKernelRegistrar g_cpu_kernel_##OPNAME##_reg(#OPNAME, ATTR,               \
                                                              []() { return std::make_shared<OPCLASS>(); });

#define MS_REG_CPU_KERNEL_T(OPNAME, ATTR, OPCLASS, T)                                         \
  static_assert(std::is_base_of<CPUKernel, OPCLASS<T>>::value, " must be base of CPUKernel"); \
  static const CPUKernelRegistrar g_cpu_kernel_##OPNAME##_##T##_reg(#OPNAME, ATTR,            \
                                                                    []() { return std::make_shared<OPCLASS<T>>(); });

#define MS_REG_CPU_KERNEL_T_S(OPNAME, ATTR, OPCLASS, T, S)                                       \
  static_assert(std::is_base_of<CPUKernel, OPCLASS<T, S>>::value, " must be base of CPUKernel"); \
  static const CPUKernelRegistrar g_cpu_kernel_##OPNAME##_##T##_##S##_reg(                       \
    #OPNAME, ATTR, []() { return std::make_shared<OPCLASS<T, S>>(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_CPU_KERNEL_FACTORY_H_
