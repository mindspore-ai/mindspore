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
using PyExecuteOutputUserDataPtr = std::shared_ptr<PyExecuteOutputUserData>;

class PyExecuteCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  PyExecuteCpuKernelMod() : kernel_node_(nullptr) {}
  ~PyExecuteCpuKernelMod() = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;
  bool need_user_data() const override { return true; }
  // User data is the extra dat-a required when the kernel is launched, It will be set before launch by runtime.
  void set_input_user_data(UserData *const user_data, size_t input_index) override {
    input_user_data_[input_index] = user_data;
  }
  void set_output_user_data(UserData *const user_data, size_t output_index) override {
    output_user_data_[output_index] = user_data;
  }

 private:
  void AttachPyOutputData(const py::object &py_res);
  CNodePtr kernel_node_{nullptr};
  std::vector<PyExecuteInputInfo> inputs_info_;
  std::map<size_t, UserData *> input_user_data_;
  std::map<size_t, UserData *> output_user_data_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYEXECUTE_KERNEL_H_
