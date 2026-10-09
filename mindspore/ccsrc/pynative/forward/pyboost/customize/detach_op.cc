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

#include <memory>
#include "include/utils/tensor_py.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/python_attr.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/runtime/lazy_fusion.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "pynative/forward/pyboost/customize/pyboost_custom_reg.h"

namespace mindspore::pynative {
namespace {
bool NeedFallback(PyObject *obj) {
  if (!fallback_enabled() || tensor::IsPyObjectPythonTensorStrict(obj)) {
    return false;
  }
  return FastGetPyObjectAttr(obj, GetFallbackStr().c_str()).ptr() != nullptr;
}
}  // namespace

static const char *OP_NAME = "Detach";
py::object PyboostDetach(const py::object &input) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, OP_NAME, false,
                                     true);
  if (!tensor::IsTensorPy(input)) {
    MS_EXCEPTION(TypeError) << OP_NAME << " input is not a tensor";
  }
  if (NeedFallback(input.ptr())) {
    auto op_call = std::make_shared<OpCall>(
      OP_NAME, [](const py::args &args, const py::kwargs &kwargs) { return PyboostDetach(args[0]); });
    py::tuple args_tuple = py::make_tuple(input);
    return py::reinterpret_steal<py::object>(pynative::HandleFallback(args_tuple.ptr(), py::cast(op_call)));
  }
  PyNativeAlgo::PyBoost::MarkSideEffect(input.ptr());
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, 1, true>();
  auto promises = tensor::TransformPromise(py_output);
  const auto input_value = tensor::ConvertToValue(input);
  DispatchOp(std::make_shared<PassthroughFrontendTask>(
    [input_value, promises]() {
      LazyFusionFlush();
      const auto &input_tensor = PyNativeAlgo::Common::StubNodeToTensor(input_value);
      auto output = std::make_shared<tensor::Tensor>(*input_tensor);
      output->set_auto_grad_meta_data(nullptr);
      tensor::SetPromise(promises, output);
    },
    [promises]() { tensor::SetException(promises); }));
  return py::reinterpret_steal<py::object>(tensor::TransformOutput(py_output));
}

}  // namespace mindspore::pynative
