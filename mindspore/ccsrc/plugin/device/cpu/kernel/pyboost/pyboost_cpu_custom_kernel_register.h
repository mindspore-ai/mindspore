/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PYBOOST_CPU_CUSTOM_KERNEL_H_
#define MINDSPORE_PYBOOST_CPU_CUSTOM_KERNEL_H_

#include <set>
#include <string>

namespace mindspore {
namespace kernel {
namespace pyboost {

class PyBoostCpuCustomKernel {
 public:
  static PyBoostCpuCustomKernel &GetInstance();

  // Register pyboost custom kernel
  void Register(const std::string &key) { custom_kernel_set_.insert(key); }

  // Check cpu custom kernel have been already registered
  bool IsPyBoostCustomRegistered(const std::string &op_name) {
    return custom_kernel_set_.find(op_name) != custom_kernel_set_.end();
  }

 private:
  std::set<std::string> custom_kernel_set_;
};

class PyBoostCpuCustomKernelRegistrar {
 public:
  explicit PyBoostCpuCustomKernelRegistrar(const std::string &name) {
    PyBoostCpuCustomKernel::GetInstance().Register(name);
  }
  ~PyBoostCpuCustomKernelRegistrar() = default;
};

#define MS_REG_PYBOOST_CPU_CUSTOM_KERNEL(NAME) \
  static const PyBoostCpuCustomKernelRegistrar g_##NAME##_pyboost_cpu_custom(#NAME);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_PYBOOST_CPU_CUSTOM_KERNEL_H_
