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

#include "plugin/device/ascend/hal/hardware/ge_deprecated_interface.h"
#include <algorithm>
#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/scoped_long_running.h"
#include "graph/model.h"
#include "transform/graph_ir/op_adapter_map.h"

using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::transform::GeTensorPtr;
using mindspore::transform::MeTensorPtr;
using mindspore::transform::Status;

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace ascend {
namespace {
void ConvertObjectToTensors(const py::dict &dict, transform::TensorOrderMap *const tensors) {
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<tensor::Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);
    if (py::isinstance<py::float_>(item.second.attr("data"))) {
      // convert float to tensor with shape([1])
      tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<py::int_>(item.second.attr("data"))) {
      // convert int64_t to tensor with shape([1])
      tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<tensor::Tensor>(item.second.attr("data"))) {
      // cast tensor
      tensor = py::cast<std::shared_ptr<tensor::Tensor>>(item.second.attr("data"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}
}  // namespace

void GeDeprecatedInterface::DoExecNonInputGraph(const std::string &phase) {
  std::vector<GeTensorPtr> ge_tensors;
  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;
  run_options.name = phase;
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return;
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    ScopedLongRunning release;
    Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph:" << run_options.name << " failed";
      return;
    }
  }
}

bool GeDeprecatedInterface::InitExecDataset(const std::string &queue_name, int64_t size, int64_t batch_size,
                                            const std::vector<TypePtr> &types,
                                            const std::vector<std::vector<int64_t>> &shapes,
                                            const std::vector<int64_t> &input_indexes, const std::string &phase) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  ge_device_context_->Initialize();
  std::vector<int64_t> ge_types;
  (void)std::transform(types.begin(), types.end(), std::back_inserter(ge_types),
                       [](const TypePtr &i) -> int64_t { return transform::ConvertDataType(i->type_id()); });

  ConfigManager::GetInstance().set_dataset_mode(DatasetMode::DS_SINK_MODE);
  ConfigManager::GetInstance().set_iter_num(queue_name, size);
  ConfigManager::GetInstance().set_dataset_phase(phase);

  DatasetGraphParam param(queue_name, size, batch_size, ge_types, shapes, input_indexes);
  ConfigManager::GetInstance().set_dataset_param(param);

  auto env_ge = common::GetEnv("MS_ENABLE_GE");
  auto env_training = common::GetEnv("MS_GE_TRAIN");
  bool training = false;
  if (env_ge == "1" && env_training == "1") {
    training = true;
  }
  if (training) {
    (void)setenv("GE_TRAIN", "1", 1);
  } else {
    (void)setenv("GE_TRAIN", "0", 1);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    if (transform::CompileDatasetGraph(param, phase) != transform::SUCCESS) {
      MS_LOG(ERROR) << "Build dateset graph failed.";
      return false;
    }

    GeDeviceResManager::CreateSessionAndGraphRunner(training);

    MS_LOG(INFO) << "DoExecNonInputGraph:" << phase;
    DoExecNonInputGraph(phase);
  }

  return true;
}

void GeDeprecatedInterface::ExportDFGraph(const std::string &file_name, const std::string &phase,
                                          const py::object &encrypt, char *key) {
  MS_LOG(DEBUG) << "Export graph begin.";
  transform::DfGraphWrapperPtr wrap_ptr = transform::GetGraphByName(phase);
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed, phase = " << phase;
    return;
  }

  transform::DfGraphPtr ge_graph = wrap_ptr->graph_ptr_;
  if (ge_graph == nullptr) {
    MS_LOG(ERROR) << "Graph is null!";
    return;
  }
  if (key != nullptr) {
    if (py::isinstance<py::none()>(encrypt)) {
      MS_LOG(ERROR) << "ERROR: encrypt is not a function";
      return;
    }
    // get model stream
    ge::Model model("", "");
    model.SetGraph(*ge_graph);
    ge::Buffer model_data;
    auto ge_ret = model.Save(model_data);
    if (ge_ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "ERROR: GE model save fail";
      return;
    }
    // convert model and key into py::bytes
    const std::string str(reinterpret_cast<char *>(model_data.GetData()), model_data.GetSize());
    py::bytes model_bytes(str);
    py::bytes key_bytes(key);

    // call python encrypt func
    py::bytes encrypted_model_stream = encrypt(model_bytes, key_bytes);
    if (encrypted_model_stream == py::none()) {
      MS_LOG(ERROR) << "ERROR: Model encrypt fail";
      return;
    }
    // save to file
    std::ofstream ofs(file_name);
    if (!ofs.is_open()) {
      MS_LOG(ERROR) << "ERROR: Open File '" << file_name << "' failed!";
      return;
    }
    ofs << std::string(encrypted_model_stream);
    ofs.close();
  } else {
    if (ge_graph->SaveToFile(file_name) != 0) {
      MS_LOG(EXCEPTION) << "Export air model failed.";
    }
  }
  MS_LOG(INFO) << "Export air model finish.";
}

FuncGraphPtr GeDeprecatedInterface::BuildDFGraph(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors);
  return GeGraphExecutor::BuildDFGraph(anf_graph, init_tensors, true);
}

void GeDeprecatedInterface::ClearGraphWrapper() { transform::DfGraphManager::GetInstance().ClearGraph(); }

void GeDeprecatedInterface::ClearOpAdapterMap() { transform::OpAdapterMap::get().clear(); }

void GeDeprecatedInterface::EraseGeResource() {
  transform::DfGraphManager::GetInstance().DeleteGraphRunner();
  transform::DfGraphManager::GetInstance().EraseAnfGraph();
  transform::DfGraphManager::GetInstance().DeleteGeSession();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
