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

#include <string>
#include <typeindex>
#include <vector>

#include "gtest/gtest.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "include/utils/pynative/dispatch/dispatch_stack.h"
#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "include/utils/python_adapter.h"

namespace py = pybind11;

namespace mindspore {
namespace pynative {
namespace {
// The guard chain hands an OpFunc to the Python guard as `func`, so OpFunc has to be bound to Python before the
// first dispatch. Production binds it when mindspore._c_expression is imported (see RegMsDispatch in
// pybind_api/pynative/dispatch_pybind.cc), and pynative tests that run earlier in the same binary import that
// module. pybind11 rejects binding one C++ type twice, so reuse that binding and bind the class here only when the
// process has none.
void RegisterOpFuncBinding() {
  if (pybind11::detail::get_type_info(std::type_index(typeid(OpFunc))) != nullptr) {
    return;
  }
  static py::module_ *op_func_module = new py::module_("_ms_ut_dispatch_types");
  py::class_<OpFunc>(*op_func_module, "OpFunc")
    .def("__call__", &OpFunc::operator())
    .def_property_readonly("name", &OpFunc::name);
}

// Records the payload it carries on every invocation. Moving this callable empties the payload, the same way moving
// a std::function does, so the recorded payloads show whether the C++ callback handed to a Python guard stayed usable.
class PayloadRecordingCallback {
 public:
  explicit PayloadRecordingCallback(std::vector<std::string> *invocations) : invocations_(invocations) {}
  PayloadRecordingCallback(const PayloadRecordingCallback &) = default;
  PayloadRecordingCallback(PayloadRecordingCallback &&other) noexcept
      : invocations_(other.invocations_), payload_(std::move(other.payload_)) {
    other.payload_.clear();
  }
  PayloadRecordingCallback &operator=(const PayloadRecordingCallback &) = default;
  PayloadRecordingCallback &operator=(PayloadRecordingCallback &&other) noexcept {
    invocations_ = other.invocations_;
    payload_ = std::move(other.payload_);
    other.payload_.clear();
    return *this;
  }
  ~PayloadRecordingCallback() = default;

  py::object operator()(const py::args &) const {
    invocations_->push_back(payload_);
    return py::none();
  }

 private:
  std::vector<std::string> *invocations_;
  std::string payload_{"ready"};
};
}  // namespace

class MsDispatchGuardChainTest : public testing::Test {
 protected:
  void SetUp() override {
    py_env_ = python_adapter::set_python_scoped();
    RegisterOpFuncBinding();
  }

  void TearDown() override { DispatchGuardStack::Get().Clear(); }

  std::shared_ptr<py::scoped_interpreter> py_env_;
};

/// Feature: ms_dispatch guard chain.
/// Description: Two nested guards dispatch one op and the guard that runs first invokes the func it received twice,
///   which is what retrying or benchmarking guards do.
/// Expectation: Both invocations reach an intact C++ callback and report the operator name of the dispatched op.
TEST_F(MsDispatchGuardChainTest, NestedGuardsCanInvokeFuncMoreThanOnce) {
  py::dict guard_namespace;
  py::exec(R"(
class RecordingGuard:
    def __init__(self):
        self.op_names = []

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        self.op_names.append(func.name)
        return func(*args)


class TwiceCallingGuard(RecordingGuard):
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        self.op_names.append(func.name)
        func(*args)
        self.op_names.append(func.name)
        return func(*args)
)",
           guard_namespace);

  py::object inner_guard = guard_namespace["RecordingGuard"]();
  py::object outer_guard = guard_namespace["TwiceCallingGuard"]();
  auto &guard_stack = DispatchGuardStack::Get();
  guard_stack.Push(inner_guard);
  guard_stack.Push(outer_guard);

  std::vector<std::string> invocations;
  py::tuple empty_tuple(0);
  py::args no_args = py::reinterpret_borrow<py::args>(empty_tuple);
  // GuardChainScope is what HandleGuardChainDispatch uses to start a chain; start it here to call the chain directly.
  GuardChainScope chain_scope;
  py::object result = detail::InvokeGuardChain(PayloadRecordingCallback(&invocations), no_args, "Add");

  EXPECT_TRUE(result.is_none());
  EXPECT_EQ(invocations, (std::vector<std::string>{"ready", "ready"}));
  EXPECT_EQ(outer_guard.attr("op_names").cast<std::vector<std::string>>(), (std::vector<std::string>{"Add", "Add"}));
  EXPECT_EQ(inner_guard.attr("op_names").cast<std::vector<std::string>>(), (std::vector<std::string>{"Add"}));
}
}  // namespace pynative
}  // namespace mindspore
