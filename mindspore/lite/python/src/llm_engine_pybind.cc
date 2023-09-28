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
#include "include/api/types.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mindspore/lite/src/extendrt/cxx_api/llm_engine/llm_engine.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/python/src/common_pybind.h"

namespace mindspore::lite {
namespace py = pybind11;

std::vector<MSTensorPtr> PyLLMEnginePredict(LLMEngine *llm_engine, const LLMReq &req,
                                            const std::vector<MSTensorPtr> &inputs_ptr) {
  if (llm_engine == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  if (!llm_engine->Predict(req, inputs, &outputs).IsOk()) {
    return {};
  }
  return MSTensorToMSTensorPtr(outputs);
}

void LLMEnginePyBind(const py::module &m) {
  (void)py::enum_<LLMRole>(m, "LLMRole_", py::arithmetic())
    .value("Prompt", LLMRole::kLLMRolePrompt)
    .value("Decoder", LLMRole::kLLMRoleDecoder);

  py::class_<LLMReq>(m, "LLMReq_")
    .def(py::init<>())
    .def_readwrite("req_id", &LLMReq::req_id)
    .def_readwrite("prompt_length", &LLMReq::prompt_length)
    .def_readwrite("prompt_cluster_id", &LLMReq::prompt_cluster_id)
    .def_readwrite("decoder_cluster_id", &LLMReq::decoder_cluster_id);

  py::class_<LLMEngineStatus>(m, "LLMEngineStatus_")
    .def(py::init<>())
    .def_readwrite("empty_max_prompt_kv", &LLMEngineStatus::empty_max_prompt_kv);

  (void)py::class_<LLMEngine, std::shared_ptr<LLMEngine>>(m, "LLMEngine_")
    .def(py::init<>())
    .def("init", &LLMEngine::Init, py::call_guard<py::gil_scoped_release>())
    .def("finalize", &LLMEngine::Finalize, py::call_guard<py::gil_scoped_release>())
    .def("predict", &PyLLMEnginePredict, py::call_guard<py::gil_scoped_release>())
    .def("complete_request", &LLMEngine::CompleteRequest, py::call_guard<py::gil_scoped_release>())
    .def("fetch_status", &LLMEngine::FetchStatus, py::call_guard<py::gil_scoped_release>());
}
}  // namespace mindspore::lite
