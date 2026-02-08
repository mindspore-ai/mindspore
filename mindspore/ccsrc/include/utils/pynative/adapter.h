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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ADAPTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ADAPTER_H_

#include <functional>
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "include/utils/visible.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/callback_handler.h"

namespace py = pybind11;
namespace mindspore {
namespace pipeline {
class Resource;
using ResourcePtr = std::shared_ptr<Resource>;
}  // namespace pipeline
namespace pynative {
class COMMON_EXPORT PyNativeAdapter {
  PyNativeAdapter();
  ~PyNativeAdapter();
  HANDLER_DEFINE(bool, GetJitBpropGraph, const pipeline::ResourcePtr &, const std::string &);
  HANDLER_DEFINE(py::object, GradJit, const py::args &);
  HANDLER_DEFINE(void, SetGraphPhase, const std::string &);
};

class COMMON_EXPORT HookAdapter {
 public:
  HookAdapter();
  ~HookAdapter();
  HANDLER_DEFINE(uint64_t, RegisterTensorBackwardHook, const tensor::TensorPtr &, const py::function &);
  HANDLER_DEFINE(void, RemoveTensorBackwardHook, uint64_t);
  HANDLER_DEFINE(py::list, GetHooks, const tensor::TensorPtr &);
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ADAPTER_H_
