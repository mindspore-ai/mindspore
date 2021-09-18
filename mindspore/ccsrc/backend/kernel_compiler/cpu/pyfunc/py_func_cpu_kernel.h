/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYFUNC_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYFUNC_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"

namespace py = pybind11;
namespace mindspore {
namespace kernel {
// Indicate Python object type. The input/output of PyFun should be either Scalar or Numpy Array.
enum class PythonOjectType : char { kScalar, kNumpyArray };
// Indicate PyFunc input/output information
struct PyFuncArgumentInfo {
  // Empty vector indicate the Python object is Scalar and non-empty means Numpy Array.
  std::vector<std::vector<int64_t>> shapes;
  // Data type as int, float, bool.
  std::vector<TypePtr> dtypes;
  // Python object type
  std::vector<PythonOjectType> object_types;
};

class PyFuncCpuKernel : public CPUKernel {
 public:
  PyFuncCpuKernel() : init_(false), func_id_(-1) {}
  ~PyFuncCpuKernel() = default;

  // Init kernel including analyse PyFunc input and output info.
  void InitKernel(const CNodePtr &kernel_node) override;
  // Construct arguments with raw memory, invoke Python function and then convert result to raw memory.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  // Analyse PyFunc input/output spec.
  void BuildFuncInfo(const CNodePtr &kernel_node);
  // Get Python function from anchor.
  py::function GetPythonFunc(const int64_t &func_id);
  bool ExecuteKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  bool init_;
  // The Python object is not acceptable for `Primitive` attribute. So we pass an unique key instead of Python function.
  // ME store the Python function to a dict, and pass the key to backend kernel.
  // The kernel get the Python functhon by the key from the dict when the kernel is first invoked.
  int64_t func_id_;
  py::function py_func_;
  // Input and output specifications.
  PyFuncArgumentInfo input_infos_;
  PyFuncArgumentInfo output_infos_;
  // The kernel hold the input tensors during execution to avoid dynamic malloc/free host memory.
  std::vector<std::shared_ptr<tensor::Tensor>> input_tensors_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PYFUNC_KERNEL_H_
