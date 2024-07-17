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
#include "include/api/model.h"
#include "include/api/model_group.h"
#include "include/api/model_parallel_runner.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/python/src/common_pybind.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace mindspore::lite {
namespace py = pybind11;

std::vector<MSTensorPtr> PyModelPredict(Model *model, const std::vector<MSTensorPtr> &inputs_ptr,
                                        const std::vector<MSTensorPtr> &outputs_ptr) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  if (!outputs_ptr.empty()) {
    outputs = MSTensorPtrToMSTensor(outputs_ptr);
  }
  if (!model->Predict(inputs, &outputs).IsOk()) {
    return {};
  }
  if (!outputs_ptr.empty()) {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs_ptr[i]->SetShape(outputs[i].Shape());
      outputs_ptr[i]->SetDataType(outputs[i].DataType());
    }
    return outputs_ptr;
  }
  return MSTensorToMSTensorPtr(outputs);
}

Status PyModelResize(Model *model, const std::vector<MSTensorPtr> &inputs_ptr,
                     const std::vector<std::vector<int64_t>> &new_shapes) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return kLiteError;
  }
  auto inputs = MSTensorPtrToMSTensor(inputs_ptr);
  return model->Resize(inputs, new_shapes);
}

Status PyModelUpdateConfig(Model *model, const std::string &key, const std::map<std::string, std::string> &value) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return kLiteError;
  }
  for (auto &item : value) {
    if (model->UpdateConfig(key, item).IsError()) {
      MS_LOG(ERROR) << "Update config failed, section: " << key << ", config name: " << item.first
                    << ", config value: " << item.second;
      return kLiteError;
    }
  }
  return kSuccess;
}

std::vector<MSTensorPtr> PyModelGetInputs(Model *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetInputs());
}

std::vector<MSTensorPtr> PyModelGetOutputs(Model *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetOutputs());
}

std::string PyModelGetModelInfo(Model *model, const std::string &key) {
  std::string empty;
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return empty;
  }
  return model->GetModelInfo(key);
}

Status PyModelUpdateWeights(Model *model, const std::vector<std::vector<MSTensorPtr>> &weights) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<std::vector<MSTensor>> new_weights;
  for (auto &weight : weights) {
    std::vector<MSTensor> new_weight = MSTensorPtrToMSTensor(weight);
    new_weights.push_back(new_weight);
  }
  if (!model->UpdateWeights(new_weights).IsOk()) {
    return kLiteError;
  }
  return kSuccess;
}

void ModelPyBind(const py::module &m) {
  (void)py::enum_<ModelType>(m, "ModelType")
    .value("kMindIR", ModelType::kMindIR)
    .value("kMindIR_Lite", ModelType::kMindIR_Lite);

  (void)py::enum_<StatusCode>(m, "StatusCode")
    .value("kSuccess", StatusCode::kSuccess)
    .value("kLiteError", StatusCode::kLiteError)
    .value("kLiteNullptr", StatusCode::kLiteNullptr)
    .value("kLiteParamInvalid", StatusCode::kLiteParamInvalid)
    .value("kLiteNoChange", StatusCode::kLiteNoChange)
    .value("kLiteSuccessExit", StatusCode::kLiteSuccessExit)
    .value("kLiteMemoryFailed", StatusCode::kLiteMemoryFailed)
    .value("kLiteNotSupport", StatusCode::kLiteNotSupport)
    .value("kLiteThreadPoolError", StatusCode::kLiteThreadPoolError)
    .value("kLiteUninitializedObj", StatusCode::kLiteUninitializedObj)
    .value("kLiteFileError", StatusCode::kLiteFileError)
    .value("kLiteServiceDeny", StatusCode::kLiteServiceDeny)
    .value("kLiteOutOfTensorRange", StatusCode::kLiteOutOfTensorRange)
    .value("kLiteInputTensorError", StatusCode::kLiteInputTensorError)
    .value("kLiteReentrantError", StatusCode::kLiteReentrantError)
    .value("kLiteGraphFileError", StatusCode::kLiteGraphFileError)
    .value("kLiteNotFindOp", StatusCode::kLiteNotFindOp)
    .value("kLiteInvalidOpName", StatusCode::kLiteInvalidOpName)
    .value("kLiteInvalidOpAttr", StatusCode::kLiteInvalidOpAttr)
    .value("kLiteOpExecuteFailure", StatusCode::kLiteOpExecuteFailure)
    .value("kLiteFormatError", StatusCode::kLiteFormatError)
    .value("kLiteInferError", StatusCode::kLiteInferError)
    .value("kLiteInferInvalid", StatusCode::kLiteInferInvalid)
    .value("kLiteInputParamInvalid", StatusCode::kLiteInputParamInvalid)
    .value("kLiteLLMKVCacheNotExist", StatusCode::kLiteLLMKVCacheNotExist)
    .value("kLiteLLMWaitProcessTimeOut", StatusCode::kLiteLLMWaitProcessTimeOut)
    .value("kLiteLLMRepeatRequest", StatusCode::kLiteLLMRepeatRequest)
    .value("kLiteLLMRequestAlreadyCompleted", StatusCode::kLiteLLMRequestAlreadyCompleted)
    .value("kLiteLLMEngineFinalized", StatusCode::kLiteLLMEngineFinalized)
    .value("kLiteLLMNotYetLink", StatusCode::kLiteLLMNotYetLink)
    .value("kLiteLLMAlreadyLink", StatusCode::kLiteLLMAlreadyLink)
    .value("kLiteLLMLinkFailed", StatusCode::kLiteLLMLinkFailed)
    .value("kLiteLLMUnlinkFailed", StatusCode::kLiteLLMUnlinkFailed)
    .value("kLiteLLMNofiryPromptUnlinkFailed", StatusCode::kLiteLLMNofiryPromptUnlinkFailed)
    .value("kLiteLLMClusterNumExceedLimit", StatusCode::kLiteLLMClusterNumExceedLimit)
    .value("kLiteLLMProcessingLink", StatusCode::kLiteLLMProcessingLink)
    .value("kLiteLLMOutOfMemory", StatusCode::kLiteLLMOutOfMemory)
    .value("kLiteLLMPrefixAlreadyExist", StatusCode::kLiteLLMPrefixAlreadyExist)
    .value("kLiteLLMPrefixNotExist", StatusCode::kLiteLLMPrefixNotExist)
    .value("kLiteLLMSeqLenOverLimit", StatusCode::kLiteLLMSeqLenOverLimit)
    .value("kLiteLLMNoFreeBlock", StatusCode::kLiteLLMNoFreeBlock)
    .value("kLiteLLMBlockOutOfMemory", StatusCode::kLiteLLMBlockOutOfMemory);

  (void)py::class_<Status, std::shared_ptr<Status>>(m, "Status")
    .def(py::init<>())
    .def(py::init<StatusCode>())
    .def("ToString", &Status::ToString)
    .def("IsOk", &Status::IsOk)
    .def("IsError", &Status::IsError)
    .def("StatusCode", &Status::StatusCode);

  (void)py::class_<Model, std::shared_ptr<Model>>(m, "ModelBind")
    .def(py::init<>())
    .def("build_from_buff",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &>(&Model::Build),
         py::call_guard<py::gil_scoped_release>())
    .def("build_from_file",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &>(&Model::Build),
         py::call_guard<py::gil_scoped_release>())
    .def("build_from_buff_with_decrypt",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("build_from_file_with_decrypt",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("load_config", py::overload_cast<const std::string &>(&Model::LoadConfig))
    .def("update_config", &PyModelUpdateConfig)
    .def("resize", &PyModelResize)
    .def("predict", &PyModelPredict, py::call_guard<py::gil_scoped_release>())
    .def("update_weights", &PyModelUpdateWeights, py::call_guard<py::gil_scoped_release>())
    .def("get_inputs", &PyModelGetInputs)
    .def("get_outputs", &PyModelGetOutputs)
    .def("get_model_info", &PyModelGetModelInfo)
    .def("get_input_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetInputByTensorName(tensor_name); })
    .def("get_output_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetOutputByTensorName(tensor_name); });
}

#ifdef PARALLEL_INFERENCE
std::vector<MSTensorPtr> PyModelParallelRunnerPredict(ModelParallelRunner *runner,
                                                      const std::vector<MSTensorPtr> &inputs_ptr,
                                                      const std::vector<MSTensorPtr> &outputs_ptr,
                                                      const MSKernelCallBack &before = nullptr,
                                                      const MSKernelCallBack &after = nullptr) {
  if (runner == nullptr) {
    MS_LOG(ERROR) << "ModelParallelRunner object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  if (!outputs_ptr.empty()) {
    outputs = MSTensorPtrToMSTensor(outputs_ptr);
  }
  if (!runner->Predict(inputs, &outputs, before, after).IsOk()) {
    return {};
  }
  return MSTensorToMSTensorPtr(outputs);
}

std::vector<MSTensorPtr> PyModelParallelRunnerGetInputs(ModelParallelRunner *runner) {
  if (runner == nullptr) {
    MS_LOG(ERROR) << "ModelParallelRunner object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(runner->GetInputs());
}

std::vector<MSTensorPtr> PyModelParallelRunnerGetOutputs(ModelParallelRunner *runner) {
  if (runner == nullptr) {
    MS_LOG(ERROR) << "ModelParallelRunner object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(runner->GetOutputs());
}
#endif

void ModelParallelRunnerPyBind(const py::module &m) {
#ifdef PARALLEL_INFERENCE
  (void)py::class_<RunnerConfig, std::shared_ptr<RunnerConfig>>(m, "RunnerConfigBind")
    .def(py::init<>())
    .def("set_config_info", py::overload_cast<const std::string &, const std::map<std::string, std::string> &>(
                              &RunnerConfig::SetConfigInfo))
    .def("get_config_info", &RunnerConfig::GetConfigInfo)
    .def("set_config_path", py::overload_cast<const std::string &>(&RunnerConfig::SetConfigPath))
    .def("get_config_path", &RunnerConfig::GetConfigPath)
    .def("set_workers_num", &RunnerConfig::SetWorkersNum)
    .def("get_workers_num", &RunnerConfig::GetWorkersNum)
    .def("set_context", &RunnerConfig::SetContext)
    .def("get_context", &RunnerConfig::GetContext)
    .def("set_device_ids", &RunnerConfig::SetDeviceIds)
    .def("get_device_ids", &RunnerConfig::GetDeviceIds)
    .def("get_context_info",
         [](RunnerConfig &runner_config) {
           const auto &context = runner_config.GetContext();
           std::string result = "thread num: " + std::to_string(context->GetThreadNum()) +
                                ", bind mode: " + std::to_string(context->GetThreadAffinityMode());
           return result;
         })
    .def("get_config_info_string", [](RunnerConfig &runner_config) {
      std::string result = "";
      const auto &config_info = runner_config.GetConfigInfo();
      for (auto &section : config_info) {
        result += section.first + ": ";
        for (auto &config : section.second) {
          auto temp = config.first + " " + config.second + "\n";
          result += temp;
        }
      }
      return result;
    });

  (void)py::class_<ModelParallelRunner, std::shared_ptr<ModelParallelRunner>>(m, "ModelParallelRunnerBind")
    .def(py::init<>())
    .def("init",
         py::overload_cast<const std::string &, const std::shared_ptr<RunnerConfig> &>(&ModelParallelRunner::Init),
         py::call_guard<py::gil_scoped_release>())
    .def("get_inputs", &PyModelParallelRunnerGetInputs)
    .def("get_outputs", &PyModelParallelRunnerGetOutputs)
    .def("predict", &PyModelParallelRunnerPredict, py::call_guard<py::gil_scoped_release>());
#endif
}

Status PyModelGroupAddModelByObject(ModelGroup *model_group, const std::vector<Model *> &models_ptr) {
  if (model_group == nullptr) {
    MS_LOG(ERROR) << "Model group object cannot be nullptr";
    return {};
  }
  std::vector<Model> models;
  for (auto model_ptr : models_ptr) {
    if (model_ptr == nullptr) {
      MS_LOG(ERROR) << "Model object cannot be nullptr";
      return {};
    }
    models.push_back(*model_ptr);
  }
  return model_group->AddModel(models);
}

void ModelGroupPyBind(const py::module &m) {
  (void)py::enum_<ModelGroupFlag>(m, "ModelGroupFlag")
    .value("kShareWeight", ModelGroupFlag::kShareWeight)
    .value("kShareWorkspace", ModelGroupFlag::kShareWorkspace)
    .value("kShareWeightAndWorkspace", ModelGroupFlag::kShareWeightAndWorkspace);

  (void)py::class_<ModelGroup, std::shared_ptr<ModelGroup>>(m, "ModelGroupBind")
    .def(py::init<ModelGroupFlag>())
    .def("add_model", py::overload_cast<const std::vector<std::string> &>(&ModelGroup::AddModel))
    .def("add_model_by_object", &PyModelGroupAddModelByObject)
    .def("cal_max_size_of_workspace",
         py::overload_cast<ModelType, const std::shared_ptr<Context> &>(&ModelGroup::CalMaxSizeOfWorkspace));
}
}  // namespace mindspore::lite
