/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/utils/pynative/dispatch/dispatch_stack.h"
#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace pynative {

/// \brief Python binding helper class for DispatchGuardStack.
/// Uses Pipeline::WaitFrontend() to avoid multi-threading conflicts.
class DispatchGuardStackPyBridge {
 public:
  static void Push(const py::object &guard) {
    // Avoid multi-threading conflicts.
    // Stack is Push/Pop in Python thread, but read in C++ frontend queue.
    runtime::Pipeline::Get().WaitFrontend();
    DispatchGuardStack::Get().Push(guard);
  }
  static void Pop() {
    runtime::Pipeline::Get().WaitFrontend();
    DispatchGuardStack::Get().Pop();
  }
  static bool HasDispatchGuard() { return DispatchGuardStack::Get().HasDispatchGuard(); }
  static size_t Depth() { return DispatchGuardStack::Get().Depth(); }
  static void Clear() { DispatchGuardStack::Get().Clear(); }
};

/// \brief Context manager to temporarily disable ms_dispatch.
/// Uses Pipeline::WaitFrontend() to avoid multi-threading conflicts.
class DisableMsDispatchGuard {
 public:
  DisableMsDispatchGuard() {}
  DisableMsDispatchGuard &__enter__() {
    runtime::Pipeline::Get().WaitFrontend();
    SetMsDispatchEnabled(false);
    return *this;
  }
  bool __exit__(py::object /*exc_type*/, py::object /*exc*/, py::object /*tb*/) {
    runtime::Pipeline::Get().WaitFrontend();
    SetMsDispatchEnabled(true);
    return false;  // rethrow exception.
  }
};

void RegMsDispatch(py::module *m) {
  // Register _DisableMsDispatchMode
  py::class_<DisableMsDispatchGuard>(*m, "_DisableMsDispatchMode")
    .def(py::init<>())
    .def("__enter__", &DisableMsDispatchGuard::__enter__, py::return_value_policy::reference_internal)
    .def("__exit__", &DisableMsDispatchGuard::__exit__);

  // Register OpFunc
  py::class_<OpFunc>(*m, "OpFunc").def("__call__", &OpFunc::operator()).def_property_readonly("name", &OpFunc::name);

  // Register _DispatchGuardStack
  py::class_<DispatchGuardStackPyBridge>(*m, "_DispatchGuardStack")
    .def_static("push", &DispatchGuardStackPyBridge::Push)
    .def_static("pop", &DispatchGuardStackPyBridge::Pop)
    .def_static("has_dispatch_guard", &DispatchGuardStackPyBridge::HasDispatchGuard)
    .def_static("depth", &DispatchGuardStackPyBridge::Depth)
    .def_static("clear", &DispatchGuardStackPyBridge::Clear);
}

}  // namespace pynative
}  // namespace mindspore
