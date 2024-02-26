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
#include "extendrt/cxx_api/llm_engine/llm_engine.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/python/src/common_pybind.h"

namespace mindspore::lite {
namespace py = pybind11;

std::pair<std::vector<MSTensorPtr>, Status> PyLLMModelPredict(LLMModel *llm_model, const LLMReq &req,
                                                              const std::vector<MSTensorPtr> &inputs_ptr) {
  if (llm_model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  auto status = llm_model->Predict(req, inputs, &outputs);
  if (!status.IsOk()) {
    return {{}, status};
  }
  return {MSTensorToMSTensorPtr(outputs), status};
}

std::pair<std::vector<MSTensorPtr>, Status> PyLLMModelPredictBatch(LLMModel *llm_model, const std::vector<LLMReq> &req,
                                                                   const std::vector<MSTensorPtr> &inputs_ptr) {
  if (llm_model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  auto status = llm_model->Predict(req, inputs, &outputs);
  if (!status.IsOk()) {
    return {{}, status};
  }
  return {MSTensorToMSTensorPtr(outputs), status};
}

Status PyLLMModelPreloadPromptPrefix(LLMModel *llm_model, const LLMReq &req,
                                     const std::vector<MSTensorPtr> &inputs_ptr) {
  if (llm_model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return kLiteError;
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  return llm_model->PreloadPromptPrefix(req, inputs);
}

std::pair<Status, std::vector<Status>> PyLLMEngineLinkClusters(LLMEngine *llm_engine,
                                                               const std::vector<LLMClusterInfo> &clusters,
                                                               int32_t timeout) {
  if (llm_engine == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object cannot be nullptr";
    return {kLiteError, {}};
  }
  std::vector<Status> rets;
  auto ret = llm_engine->LinkClusters(clusters, &rets, timeout);
  return {ret, rets};
}

std::pair<Status, std::vector<Status>> PyLLMEngineUnlinkClusters(LLMEngine *llm_engine,
                                                                 const std::vector<LLMClusterInfo> &clusters,
                                                                 int32_t timeout) {
  if (llm_engine == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object cannot be nullptr";
    return {kLiteError, {}};
  }
  std::vector<Status> rets;
  auto ret = llm_engine->UnlinkClusters(clusters, &rets, timeout);
  return {ret, rets};
}

void PyLLMClusterAppendLocalIpInfo(LLMClusterInfo *cluster_info, uint32_t ip, uint16_t port) {
  if (cluster_info == nullptr) {
    MS_LOG(ERROR) << "LLMClusterInfo object cannot be nullptr";
    return;
  }
  LLMIpInfo ip_info;
  ip_info.ip = ip;
  ip_info.port = port;
  cluster_info->local_ip_infos.push_back(ip_info);
}

std::vector<std::pair<uint32_t, uint16_t>> PyLLMClusterGetLocalIpInfo(LLMClusterInfo *cluster_info) {
  if (cluster_info == nullptr) {
    MS_LOG(ERROR) << "LLMClusterInfo object cannot be nullptr";
    return {};
  }
  std::vector<std::pair<uint32_t, uint16_t>> ip_infos;
  auto &local_ip_infos = cluster_info->local_ip_infos;
  (void)std::transform(local_ip_infos.begin(), local_ip_infos.end(), std::back_inserter(ip_infos),
                       [](auto &item) { return std::make_pair(item.ip, item.port); });
  return ip_infos;
}

void PyLLMClusterAppendRemoteIpInfo(LLMClusterInfo *cluster_info, uint32_t ip, uint16_t port) {
  if (cluster_info == nullptr) {
    MS_LOG(ERROR) << "LLMClusterInfo object cannot be nullptr";
    return;
  }
  LLMIpInfo ip_info;
  ip_info.ip = ip;
  ip_info.port = port;
  cluster_info->remote_ip_infos.push_back(ip_info);
}

std::vector<std::pair<uint32_t, uint16_t>> PyLLMClusterGetRemoteIpInfo(LLMClusterInfo *cluster_info) {
  if (cluster_info == nullptr) {
    MS_LOG(ERROR) << "LLMClusterInfo object cannot be nullptr";
    return {};
  }
  std::vector<std::pair<uint32_t, uint16_t>> ip_infos;
  auto &remote_ip_infos = cluster_info->remote_ip_infos;
  (void)std::transform(remote_ip_infos.begin(), remote_ip_infos.end(), std::back_inserter(ip_infos),
                       [](auto &item) { return std::make_pair(item.ip, item.port); });
  return ip_infos;
}

std::pair<Status, std::shared_ptr<LLMModel>> PyLLMEngineAddModel(LLMEngine *llm_engine,
                                                                 const std::vector<std::string> &model_paths,
                                                                 const std::map<std::string, std::string> &options,
                                                                 const std::string &postprocess_model_path) {
  if (llm_engine == nullptr) {
    MS_LOG(ERROR) << "LLMClusterInfo object cannot be nullptr";
    return {kLiteError, nullptr};
  }
  auto llm_model = std::make_shared<LLMModel>();
  if (llm_model == nullptr) {
    MS_LOG(ERROR) << "Failed to create LLMModel object";
    return {kLiteError, nullptr};
  }
  auto status = llm_engine->AddModel(llm_model.get(), model_paths, options, postprocess_model_path);
  return {status, llm_model};
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
    .def_readwrite("decoder_cluster_id", &LLMReq::decoder_cluster_id)
    .def_readwrite("prefix_id", &LLMReq::prefix_id);

  py::class_<LLMClusterInfo>(m, "LLMClusterInfo_")
    .def(py::init<>())
    .def_readwrite("remote_cluster_id", &LLMClusterInfo::remote_cluster_id)
    .def_readwrite("remote_role_type", &LLMClusterInfo::remote_role_type)
    .def("append_local_ip_info", &PyLLMClusterAppendLocalIpInfo)
    .def("append_remote_ip_info", &PyLLMClusterAppendRemoteIpInfo)
    .def("get_local_ip_infos", &PyLLMClusterGetLocalIpInfo)
    .def("get_remote_ip_infos", &PyLLMClusterGetRemoteIpInfo);

  py::class_<LLMTensorInfo>(m, "LLMTensorInfo_")
    .def(py::init<>())
    .def_readwrite("name", &LLMTensorInfo::name)
    .def_readwrite("shape", &LLMTensorInfo::shape)
    .def_readwrite("dtype", &LLMTensorInfo::dtype);

  py::class_<LLMEngineStatus>(m, "LLMEngineStatus_")
    .def(py::init<>())
    .def_readwrite("empty_max_prompt_kv", &LLMEngineStatus::empty_max_prompt_kv);

  (void)py::class_<LLMModel, std::shared_ptr<LLMModel>>(m, "LLMModel_")
    .def(py::init<>())
    .def("predict", &PyLLMModelPredict, py::call_guard<py::gil_scoped_release>())
    .def("predict_batch", &PyLLMModelPredictBatch, py::call_guard<py::gil_scoped_release>())
    .def("preload_prompt_prefix", &PyLLMModelPreloadPromptPrefix, py::call_guard<py::gil_scoped_release>())
    .def("release_prompt_prefix", &LLMModel::ReleasePromptPrefix, py::call_guard<py::gil_scoped_release>())
    .def("pull_kv", &LLMModel::PullKV, py::call_guard<py::gil_scoped_release>())
    .def("merge_kv", &LLMModel::MergeKV, py::call_guard<py::gil_scoped_release>())
    .def("get_input_infos", &LLMModel::GetInputInfos);

  (void)py::class_<LLMEngine, std::shared_ptr<LLMEngine>>(m, "LLMEngine_")
    .def(py::init<LLMRole, uint64_t, const std::string &>())
    .def("add_model", &PyLLMEngineAddModel, py::call_guard<py::gil_scoped_release>())
    .def("init", &LLMEngine::Init, py::call_guard<py::gil_scoped_release>())
    .def("finalize", &LLMEngine::Finalize, py::call_guard<py::gil_scoped_release>())
    .def("fetch_status", &LLMEngine::FetchStatus, py::call_guard<py::gil_scoped_release>())
    .def("link_clusters", &PyLLMEngineLinkClusters, py::call_guard<py::gil_scoped_release>())
    .def("unlink_clusters", &PyLLMEngineUnlinkClusters, py::call_guard<py::gil_scoped_release>())
    .def("complete_request", &LLMEngine::CompleteRequest, py::call_guard<py::gil_scoped_release>());
}
}  // namespace mindspore::lite
