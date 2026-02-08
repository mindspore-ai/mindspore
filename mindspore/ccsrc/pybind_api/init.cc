/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include <pybind11/operators.h>
#include <memory>
#include <string>
#include "pybind_api/backend/backend_api.h"
#include "pybind_api/frontend/frontend_api.h"
#include "pybind_api/ops/ops_api.h"
#include "pybind_api/parallel/parallel_api.h"
#include "pybind_api/pynative/pynative_api.h"
#include "pybind_api/runtime/runtime_api.h"
#include "pybind_api/tools/tools_api.h"
#include "pybind_api/utils/utils_api.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pybind_api/frontend/manager.h"
#include "utils/ms_context.h"
#include "include/utils/python_adapter.h"
#ifdef _WIN32
#include "kernel/cpu/utils/cpu_utils.h"
#endif

namespace py = pybind11;
namespace mindspore {
void RegModule(py::module *m) {
  RegFrontendModule(m);

  RegPyNativeModule(m);

  RegParallelModule(m);

  RegOpsModule(m);

  RegBackendModule(m);

  RegRuntimeModule(m);

  RegToolsModule(m);

  RegUtilsModule(m);
}

void RegisterExit() {
  (void)py::module::import("atexit").attr("register")(py::cpp_function{[&]() -> void {
    MS_LOG(INFO) << "Start register...";
    mindspore::MsContext::GetInstance()->RegisterCheckEnv(nullptr);
    mindspore::MsContext::GetInstance()->RegisterSetEnv(nullptr);
    MS_LOG(INFO) << "Start mindspore.profiler...";
    try {
      auto profiler = py::module::import("mindspore.profiler").attr("EnvProfiler")();
      (void)profiler.attr("analyse")();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Failed to parse profiler data." << e.what();
    }

#ifdef ENABLE_MINDDATA
    MS_LOG(INFO) << "Start releasing dataset handles...";
    py::module iterators = py::module::import("mindspore.dataset.engine.iterators");
    (void)iterators.attr("_cleanup")();
    MS_LOG(INFO) << "End release dataset handles.";
#endif

    // only in case that c++ calling python interface, ClearResAtexit should be called.
    if (mindspore::python_adapter::IsPythonEnv()) {
      mindspore::ClearResAtexit();
    }
  }});
}

void RegModuleHelper(py::module *m) {
  static std::once_flag onlyCalledOnce;
  std::call_once(onlyCalledOnce, RegModule, m);
}
}  // namespace mindspore

// Interface with python
PYBIND11_MODULE(_c_expression, m) {
#ifdef _WIN32
  // Use dummy function to force link to mindspore_ops_cpu on windows
  mindspore::kernel::ForceLinkOpsHost();
#endif
  // The OMP_NUM_THREADS has no effect when set in backend, so set it here in advance.
  mindspore::common::SetOMPThreadNum();

  m.doc() = "MindSpore c plugin";

  mindspore::RegModuleHelper(&m);

  mindspore::ScopedLongRunning::SetHook(std::make_unique<mindspore::GilScopedLongRunningHook>());

  mindspore::RegisterExit();
}
