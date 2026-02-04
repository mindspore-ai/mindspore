/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_CUSTOM_KERNEL_MOD_IMPL_PY_FUNC_KERNEL_MOD_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_CUSTOM_KERNEL_MOD_IMPL_PY_FUNC_KERNEL_MOD_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>

#include "ops/op_def.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore::kernel {

class PyFuncKernelMod : public KernelMod {
 public:
  PyFuncKernelMod() = default;
  ~PyFuncKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &output, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in PyFunc kernel.";
  }
  void set_fullname(const std::string &fullname) override { fullname_ = fullname; }

 protected:
  pybind11::function GetPythonFunc() const;
  pybind11::tuple PreprocessInputs(const std::vector<KernelTensor *> &inputs);
  bool PostprocessOutputs(pybind11::handle result, const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  // The Python object is not acceptable for `Primitive` attribute. So we pass an unique key instead of Python function.
  // mindspore.ops.operations.PyFunc store the Python function to a dict, and pass the key to backend kernel.
  // The kernel get the Python functhon by the key from the dict when the kernel is first invoked.
  int64_t func_id_{0};
  pybind11::function py_func_;
  bool init_{false};
  std::string fullname_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_CUSTOM_KERNEL_MOD_IMPL_PY_FUNC_KERNEL_MOD_H_
