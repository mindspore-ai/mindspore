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
  py::enum_<ModelType>(m, "ModelType")
    .value("kMindIR", ModelType::kMindIR)
    .value("kMindIR_Lite", ModelType::kMindIR_Lite);

  py::class_<Model, std::shared_ptr<Model>>(m, "ModelBind")
    .def(py::init<>())
    .def("build_from_buff",
         [](Model *model, const void *model_data, size_t data_size, ModelType model_type,
            const std::shared_ptr<Context> &model_context = nullptr) {
           auto ret = model->Build(model_data, data_size, model_type, model_context);
           return static_cast<uint32_t>(ret.StatusCode());
         })
    .def("build_from_file",
         [](Model *model, const std::string &model_path, ModelType model_type,
            const std::shared_ptr<Context> &context = nullptr) {
           auto ret = model->Build(model_path, ModelType::kMindIR_Lite, context);
           return static_cast<uint32_t>(ret.StatusCode());
         })
    .def("resize",
         [](Model *model, const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
           auto ret = model->Resize(inputs, dims);
           return static_cast<uint32_t>(ret.StatusCode());
         })
    .def("predict",
         [](Model *model, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
            const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr) {
           auto ret = model->Predict(inputs, outputs, before, after);
           return static_cast<uint32_t>(ret.StatusCode());
         })
    .def("get_inputs", &Model::GetInputs)
    .def("get_outputs", &Model::GetOutputs)
    .def("get_input_by_tensor_name",
         [](Model *model, const std::string &tensor_name) { return model->GetInputByTensorName(tensor_name); })
    .def("get_output_by_tensor_name",
         [](Model *model, const std::string &tensor_name) { return model->GetOutputByTensorName(tensor_name); });

#ifdef PARALLEL_INFERENCE
  py::class_<ModelParallelRunner, std::shared_ptr<ModelParallelRunner>>(m, "ModelParallelRunnerBind")
    .def(py::init<>([](const std::string &model_path, const std::shared_ptr<Context> &context, int workers_num) {
      auto config = std::make_shared<RunnerConfig>();
      config->context = context;
      config->workers_num = workers_num;
      auto runner = std::make_shared<ModelParallelRunner>();
      auto ret = runner->Init(model_path, config);
      if (ret.StatusCode() != kSuccess) {
        std::cout << "Init failed" << std::endl;
      }
      return runner;
    }))
    .def("get_inputs", &ModelParallelRunner::GetInputs)
    .def("get_outputs", &ModelParallelRunner::GetOutputs)
    .def("predict", [](ModelParallelRunner &model, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                       const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr) {
      auto ret = model.Predict(inputs, outputs, before, after);
      return static_cast<uint32_t>(ret.StatusCode());
    });
#endif
}
}  // namespace mindspore::lite
