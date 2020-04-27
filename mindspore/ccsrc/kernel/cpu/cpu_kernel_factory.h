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
#include "common/utils.h"
#include "kernel/cpu/cpu_kernel.h"
namespace mindspore {
namespace kernel {
using CPUKernelCreator = std::function<std::shared_ptr<CPUKernel>()>;
class CPUKernelFactory {
 public:
  static CPUKernelFactory &Get();
  void Register(const std::string &kernel_name, CPUKernelCreator &&kernel_creator);
  std::shared_ptr<CPUKernel> Create(const std::string &kernel_name);

 private:
  CPUKernelFactory() = default;
  ~CPUKernelFactory() = default;
  DISABLE_COPY_AND_ASSIGN(CPUKernelFactory)
  std::map<std::string, CPUKernelCreator> kernel_creators_;
};

class CPUKernelRegistrar {
 public:
  CPUKernelRegistrar(const std::string &kernel_name, CPUKernelCreator &&kernel_creator) {
    CPUKernelFactory::Get().Register(kernel_name, std::move(kernel_creator));
  }
  ~CPUKernelRegistrar() = default;
};

#define MS_REG_CPU_KERNEL(KERNEL_NAME, KERNEL_CLASS)                             \
  static const CPUKernelRegistrar g_cpu_kernel_##KERNEL_NAME##_reg(#KERNEL_NAME, \
                                                                   []() { return std::make_shared<KERNEL_CLASS>(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_CPU_KERNEL_FACTORY_H_
