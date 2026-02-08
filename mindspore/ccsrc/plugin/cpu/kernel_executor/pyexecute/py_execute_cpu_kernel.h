/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYEXECUTE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYEXECUTE_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "include/utils/pipeline/fallback.h"
#include "kernel/cpu/cpu_kernel.h"

namespace py = pybind11;
namespace mindspore {
namespace kernel {
class PyExecuteCpuKernelMod : public NativeCpuKernelMod {
 public:
  PyExecuteCpuKernelMod() {}
  ~PyExecuteCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override;
  bool need_user_data() const override { return true; }

 private:
  bool is_output_any_{true};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYEXECUTE_KERNEL_H_
