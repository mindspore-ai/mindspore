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
#include "include/api/model_parallel_runner.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace mindspore::lite {
namespace py = pybind11;

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
    .value("kLiteInputParamInvalid", StatusCode::kLiteInputParamInvalid);

  (void)py::class_<Status, std::shared_ptr<Status>>(m, "Status")
    .def(py::init<>())
    .def("ToString", &Status::ToString)
    .def("IsOk", &Status::IsOk)
    .def("IsError", &Status::IsError);

  (void)py::class_<Model, std::shared_ptr<Model>>(m, "ModelBind")
    .def(py::init<>())
    .def("build_from_buff",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &>(&Model::Build))
    .def("build_from_file",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &>(&Model::Build))
    .def("build_from_buff_with_decrypt",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("build_from_file_with_decrypt",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("load_config", py::overload_cast<const std::string &>(&Model::LoadConfig))
    .def("resize", &Model::Resize)
    .def("predict", py::overload_cast<const std::vector<MSTensor> &, std::vector<MSTensor> *, const MSKernelCallBack &,
                                      const MSKernelCallBack &>(&Model::Predict))
    .def("get_inputs", &Model::GetInputs)
    .def("get_outputs", &Model::GetOutputs)
    .def("get_input_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetInputByTensorName(tensor_name); })
    .def("get_output_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetOutputByTensorName(tensor_name); });
}

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
         py::overload_cast<const std::string &, const std::shared_ptr<RunnerConfig> &>(&ModelParallelRunner::Init))
    .def("get_inputs", &ModelParallelRunner::GetInputs)
    .def("get_outputs", &ModelParallelRunner::GetOutputs)
    .def("predict", [](ModelParallelRunner &runner, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                       const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr) {
      {
        py::gil_scoped_release release;
        auto status = runner.Predict(inputs, outputs, before, after);
        if (status != kSuccess) {
          std::vector<MSTensor> empty;
          return empty;
        }
        return *outputs;
      }
    });
#endif
}
}  // namespace mindspore::lite
