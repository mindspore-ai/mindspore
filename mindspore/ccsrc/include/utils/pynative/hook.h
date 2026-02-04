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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_HOOK_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_HOOK_H_
#include <functional>
#include "pybind11/pybind11.h"
#include "include/utils/visible.h"
#include "mindspore/core/include/ir/tensor.h"
#include "mindspore/core/include/ir/anf.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;
struct COMMON_EXPORT BackwardNodePreHook {
  virtual ~BackwardNodePreHook() = default;
  virtual void operator()(ValuePtrList *grad_outputs) = 0;
};

struct COMMON_EXPORT BackwardNodePostHook {
  virtual ~BackwardNodePostHook() = default;
  virtual void operator()(ValuePtrList *grad_inputs, const ValuePtrList &grad_outputs) = 0;
};

struct COMMON_EXPORT PyTensorBackwardNodePreHook : public BackwardNodePreHook {
  PyTensorBackwardNodePreHook(const py::function &hook_fn, size_t output_idx);
  ~PyTensorBackwardNodePreHook() override;
  void operator()(ValuePtrList *grad_outputs) override;
  py::function hook_fn_;
  size_t output_idx_;
};

using CppHookFn = std::function<tensor::TensorPtr(const tensor::TensorPtr &)>;
struct COMMON_EXPORT CppTensorBackwardNodePreHook : public BackwardNodePreHook {
  CppTensorBackwardNodePreHook(CppHookFn hook_fn, size_t output_idx);
  void operator()(ValuePtrList *grad) override;
  CppHookFn hook_fn_;
  size_t output_idx_;
};

struct COMMON_EXPORT PyBackwardNodePreHook : public BackwardNodePreHook {
  explicit PyBackwardNodePreHook(PyObject *hook_dict);
  ~PyBackwardNodePreHook() override;
  void operator()(ValuePtrList *grad_outputs) override;
  PyObject *hook_dict_;
};

using RetainGradHookFn = std::function<void(const tensor::TensorPtr &grad)>;
struct COMMON_EXPORT RetainGradHook {
  explicit RetainGradHook(RetainGradHookFn hook_fn);
  ~RetainGradHook() = default;
  void operator()(const ValuePtr &grad);
  RetainGradHookFn hook_fn_;
};

struct COMMON_EXPORT PyBackwardNodePostHook : public BackwardNodePostHook {
  explicit PyBackwardNodePostHook(PyObject *hook_dict);
  ~PyBackwardNodePostHook() override;
  void operator()(ValuePtrList *grad_inputs, const ValuePtrList &grad_outputs) override;
  PyObject *hook_dict_;
};
}  // namespace mindspore::pynative::autograd
#endif
