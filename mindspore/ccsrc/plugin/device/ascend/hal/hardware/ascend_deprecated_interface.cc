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

#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include <algorithm>
#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/scoped_long_running.h"
#include "graph/model.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"
#include "acl/acl_tdt.h"
#include "runtime/dev.h"
#include "runtime/config.h"
#include "toolchain/plog.h"
#include "common/util/error_manager/error_manager.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/device/distribute/ascend_collective.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"

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

void AscendDeprecatedInterface::DoExecNonInputGraph(const std::string &phase) {
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

bool AscendDeprecatedInterface::InitExecDataset(const std::string &queue_name, int64_t size, int64_t batch_size,
                                                const std::vector<TypePtr> &types,
                                                const std::vector<std::vector<int64_t>> &shapes,
                                                const std::vector<int64_t> &input_indexes, const std::string &phase) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  ge_device_context_->Initialize();
  std::vector<int64_t> ge_types;
  (void)std::transform(types.begin(), types.end(), std::back_inserter(ge_types), [](const TypePtr &i) -> int64_t {
    return static_cast<int64_t>(transform::ConvertDataType(i->type_id()));
  });

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

void AscendDeprecatedInterface::ExportDFGraph(const std::string &file_name, const std::string &phase,
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
    ::ge::Model model("", "");
    model.SetGraph(*ge_graph);
    ::ge::Buffer model_data;
    auto ge_ret = model.Save(model_data);
    if (ge_ret != ::ge::SUCCESS) {
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

FuncGraphPtr AscendDeprecatedInterface::BuildDFGraph(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors);
  return GeGraphExecutor::BuildDFGraph(anf_graph, init_tensors, true);
}

void AscendDeprecatedInterface::ClearGraphWrapper() { transform::DfGraphManager::GetInstance().ClearGraph(); }

void AscendDeprecatedInterface::ClearOpAdapterMap() { transform::OpAdapterMap::get().clear(); }

void AscendDeprecatedInterface::EraseGeResource() {
  transform::DfGraphManager::GetInstance().DeleteGraphRunner();
  transform::DfGraphManager::GetInstance().EraseAnfGraph();
  transform::DfGraphManager::GetInstance().DeleteGeSession();
}

uint32_t AscendDeprecatedInterface::InitCollective() {
#ifdef WITH_BACKEND
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_LOG(INFO) << "The process are launched with OpenMPI, the environment variable for rank table will be ignored "
                  "even if set by users.";
  MS_LOG(INFO) << "mpi collective init.";
  if (!collective::HcclCollectiveGroup::instance().InitCollective()) {
    MS_LOG(EXCEPTION) << "Mpi init failed, please check if mpirun is used correctly.";
  }
  auto rank_id = collective::HcclCollectiveGroup::instance().GetRankId(kHcclWorldGroup);
  (void)common::SetEnv(kRankID, std::to_string(rank_id).c_str());
  uint32_t device_id = IntToUint(collective::HcclCollectiveGroup::instance().GetDeviceId());
  (void)common::SetEnv("DEVICE_ID", std::to_string(device_id).c_str());
  ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  return device_id;
#else
  return 0;
#endif
}

void AscendDeprecatedInterface::DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {
  return profiler::ascend::ParallelStrategy::GetInstance()->DumpProfileParallelStrategy(func_graph);
}

bool AscendDeprecatedInterface::OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  // set MS_CTX_ENABLE_GE_HETEROGENOUS true if ge heterogeneous mode
  int32_t is_heterogeneous = 0;
  (void)rtGetIsHeterogenous(&is_heterogeneous);
  ms_context_ptr->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, is_heterogeneous == 1);

  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return true;
  }

  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) != 0) {
    MS_LOG(DEBUG) << "ACLTDT Dataset client is already opened.";
    ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
    return true;
  }

  auto role = common::GetEnv("MS_ROLE");
  if (strcmp(role.c_str(), "MS_SCHED") == 0 || strcmp(role.c_str(), "MS_PSERVER") == 0) {
    return true;
  }

  uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  uint32_t rank_size;
  auto rank_size_env = common::GetEnv("RANK_SIZE");
  if (rank_size_env.empty()) {
    MS_LOG(INFO) << "Should config rank size.";
    rank_size = 1;
  } else {
    int rank_env = std::stoi(rank_size_env);
    if (rank_env <= 0) {
      MS_LOG(EXCEPTION) << "Error rank size " << rank_env << ".";
    }
    rank_size = IntToUint(rank_env);
  }

  int log_ret = DlogReportInitialize();
  if (log_ret != 0) {
    MS_LOG(WARNING) << "Init slog failed, ret = " << log_ret;
  }

  if (ErrorManager::GetInstance().Init() != 0) {
    MS_LOG(WARNING) << "Init ascend error manager failed, some ascend error log may be left out.";
  }
  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";
  auto ret = rtSetDevice(static_cast<int32_t>(device_id));
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " call rtSetDevice failed, ret[" << static_cast<int>(ret) << "]."
                      << GetErrorMessage(true);
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
  auto thread_crt = [](const std::string &path, const acltdtChannelHandle *acl_handle) {
    return std::thread(TensorPrint(path, acl_handle));
  };
  CreateTensorPrintThread(thread_crt);
  return true;
}

bool AscendDeprecatedInterface::CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_LOG(INFO) << "Start to close tsd, ref = " << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF);
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_TSD_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_TSD_REF, 0);
    pybind11::gil_scoped_release gil_release;
    DestroyTensorPrintThread();
    if (ErrorManager::GetInstance().Init() != 0) {
      MS_LOG(WARNING) << "Init ascend error manager failed, some ascend error log may be left out.";
    }
    uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = rtDeviceReset(static_cast<int32_t>(device_id));
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call rtDeviceReset failed, ret[" << static_cast<int>(ret) << "]."
                        << GetErrorMessage(true);
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call rtDeviceReset, destroy and close tsd successful, ret[" << static_cast<int>(ret) << "]";
    (void)DlogReportFinalize();
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}

bool AscendDeprecatedInterface::IsTsdOpened(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) > 0;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
