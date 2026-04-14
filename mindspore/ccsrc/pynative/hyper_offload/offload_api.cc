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
#include "pybind_api/pynative/pynative_api.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pynative/hyper_offload/offload_manager.h"
#include "pynative/hyper_offload/hyper_offload_context.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

namespace py = pybind11;

void RegOffloadModule(py::module *m) {
  // Register HyperOffloadContext class
  (void)py::class_<HyperOffloadContext, HyperOffloadContextPtr>(*m, "HyperOffloadContext_")
    .def("start_offload", &HyperOffloadContext::StartOffload, "Start tracing operators (sync)")
    .def("end_offload", &HyperOffloadContext::EndOffload, "End tracing operators (sync)")
    .def("start_offload_async", &HyperOffloadContext::StartOffloadAsync, "Start tracing operators (async)")
    .def("end_offload_async", &HyperOffloadContext::EndOffloadAsync, "End tracing operators (async)")
    .def("reset", &HyperOffloadContext::Reset, "Reset the context")
    .def("is_active", &HyperOffloadContext::IsActive, "Check if context is active")
    .def("state", &HyperOffloadContext::state, "Get current state");

  (void)py::class_<OffloadManager, std::unique_ptr<OffloadManager, py::nodelete>>(*m, "OffloadManager")
    .def_static("get_instance", &OffloadManager::GetInstance, py::return_value_policy::reference,
                "Get the singleton instance of OffloadManager")
    .def("get_or_create_offload_ctx", &OffloadManager::GetOrCreateOffloadCtx, py::arg("offload_key"), R"(
      Get or create an HyperOffloadContext by offload_key.
      
      During warm-up phase (first step), creates a new context for the given key.
      During replay phase (subsequent steps), returns the existing context for the key.
      
      Args:
          offload_key (int): Unique key to identify the offload context.
      
      Returns:
          HyperOffloadContext: The offload context for the given key.
      
      Examples:
          >>> manager = OffloadManager.get_instance()
          >>> ctx = manager.get_or_create_offload_ctx(id(cell))
          >>> ctx.start_offload()
          >>> # ... execute forward/backward ...
          >>> ctx.end_offload()
    )")
    .def("analyse", &OffloadManager::Analyse, R"(
      Signal the end of warm-up phase and start replay phase.
      
      This should be called at the end of the first training step.
      After calling this, offload strategies will be applied in subsequent steps.
      
      Examples:
          >>> manager = OffloadManager.get_instance()
          >>> # At the end of first step
          >>> manager.analyse()
    )")
    .def_static("set_skip_offload", &OffloadManager::SetSkipOffload, py::arg("skip"),
                "Set whether to skip offload temporarily")
    .def_static("set_skip_offload_async", &OffloadManager::SetSkipOffloadAsync, py::arg("skip"),
                "Set whether to skip offload temporarily asynchronously")
    .def("reset", &OffloadManager::Reset, "Reset the OffloadManager and all contexts")
    .def_static("is_offload_ready", &OffloadManager::IsOffloadReady, "Check if offload is ready (warm-up completed)")
    .def("context_count", &OffloadManager::context_count, "Get the number of registered HyperOffloadContext instances");
}

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
