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

#include <memory>
#include <string>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace py = pybind11;
namespace mindspore {
namespace kernel {
struct PyExecuteInputInfo {
  py::object py_obj_output;
  abstract::AbstractBasePtr abstract;
  TypeId type;
  std::vector<int64_t> shape;
};

struct PyExecuteOutputUserData {
  py::object obj;
  constexpr static char key[] = "PyExecuteOutputUserData";
};

class PyExecuteCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  PyExecuteCpuKernelMod() : kernel_node_(nullptr) {}
  ~PyExecuteCpuKernelMod() = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void AttachPyOutputData(const py::object &py_res);
  py::object BuildLocalParameters(const std::vector<AddressPtr> &inputs);
  py::object BuildLocalTupleParameters(const std::vector<AddressPtr> &inputs);

  CNodePtr kernel_node_{nullptr};
  std::vector<PyExecuteInputInfo> inputs_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYEXECUTE_KERNEL_H_
