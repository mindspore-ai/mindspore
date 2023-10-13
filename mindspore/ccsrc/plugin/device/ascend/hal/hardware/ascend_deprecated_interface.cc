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
#include <tuple>
#include <utility>
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils_py.h"
#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/scoped_long_running.h"
#include "graph/model.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "toolchain/plog.h"
#include "framework/common/helper/model_helper.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"
#include "plugin/device/ascend/optimizer/enhancer/add_placeholder_for_dynamic_rnn.h"
#include "cxx_api/graph/acl/acl_env_guard.h"
#include "graph/utils/graph_utils_ex.h"
#include "mindspore/core/utils/singleton.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/device/tensorsummary_utils.h"

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
std::mutex g_tsd_mutex;
void ConvertObjectToTensors(const py::dict &dict, transform::TensorOrderMap *const tensors,
                            const FuncGraphPtr &anf_graph) {
  const auto &infer_need_update_parameter_names =
    Singleton<InferNeedUpdateParaNames>::Instance().GetInferParameterNames();
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<tensor::Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);
    bool infer = false;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool enable_ge = context_ptr->backend_policy() == "ge";
    bool is_train = false;
    if (anf_graph->has_attr("phase")) {
      std::string phase = anf_graph->get_attr("phase")->ToString();
      is_train = phase == "train";
    }
    if (enable_ge && !is_train) {
      infer = true;
    }
    if (infer && infer_need_update_parameter_names.find(name) == infer_need_update_parameter_names.end() &&
        !IsEnableRefMode()) {
      continue;
    }
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
    } else if (IsStubTensor(item.second.attr("data"))) {
      // cast stub_tensor
      tensor = ConvertStubTensor(item.second.attr("data"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

void GetInputTensor(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params,
                    std::vector<transform::GeTensorPtr> *ge_tensors) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_input_map;
  ConvertObjectToTensors(init_params, &init_input_map, anf_graph);
  std::vector<tensor::TensorPtr> init_input;
  (void)std::transform(init_input_map.begin(), init_input_map.end(), std::back_inserter(init_input),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });
  *ge_tensors = transform::ConvertInputTensors(init_input, kOpFormat_NCHW);
}
}  // namespace

void AscendDeprecatedInterface::RunInitGraph(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + anf_graph->ToString();

  auto graph_runner = transform::CheckAndGetGraphRunner(run_options);
  if (graph_runner == nullptr) {
    return;
  }

  std::vector<transform::GeTensorPtr> ge_outputs;
  std::vector<transform::GeTensorPtr> ge_tensors;
  GetInputTensor(anf_graph, init_params, &ge_tensors);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }
    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (transform::GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(INFO) << "Exec broadcast graph success.";
    }
  }
  auto &infer_need_update_parameter_names = Singleton<InferNeedUpdateParaNames>::Instance().GetInferParameterNames();
  infer_need_update_parameter_names.clear();
}

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
      MS_LOG(WARNING) << "Exec graph:" << run_options.name << " failed";
      return;
    }
  }
}

bool AscendDeprecatedInterface::InitExecDataset(const std::string &queue_name, int64_t size, int64_t batch_size,
                                                const std::vector<TypePtr> &types,
                                                const std::vector<std::vector<int64_t>> &shapes,
                                                const std::vector<int64_t> &input_indexes, const std::string &phase) {
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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    if (transform::CompileDatasetGraph(param, phase) != transform::SUCCESS) {
      MS_LOG(ERROR) << "Build dateset graph failed.";
      return false;
    }

    GeDeviceResManager::CreateSessionAndGraphRunner();

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
    model.SetGraph(::ge::GraphUtilsEx::GetComputeGraph(*ge_graph));
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
  ConvertObjectToTensors(init_params, &init_tensors, anf_graph);
  return GeGraphExecutor::BuildDFGraph(anf_graph, init_tensors, true);
}

void AscendDeprecatedInterface::ClearGraphWrapper() { transform::DfGraphManager::GetInstance().ClearGraph(); }

void AscendDeprecatedInterface::ClearOpAdapterMap() { transform::OpAdapterMap::get().clear(); }

void AscendDeprecatedInterface::DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {
  return profiler::ascend::ParallelStrategy::GetInstance()->DumpProfileParallelStrategy(func_graph);
}

bool AscendDeprecatedInterface::OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
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

  (void)ErrorManagerAdapter::Init();
  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";
  auto ret = aclrtSetDevice(static_cast<int32_t>(device_id));
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret[" << static_cast<int>(ret)
                      << "]. The details refer to 'Ascend Error Message'.";
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
  auto thread_crt = [](const std::string &path, const acltdtChannelHandle *acl_handle) {
    return std::thread(TensorPrint(path, acl_handle));
  };
  CreateTensorPrintThread(thread_crt);
  TensorSummaryUtils::GetInstance().CreateTDTSummaryThread();
  return true;
}

bool AscendDeprecatedInterface::CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
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
    TensorSummaryUtils::GetInstance().DestroyTDTSummaryThread();
    (void)ErrorManagerAdapter::Init();
    uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = aclrtResetDevice(static_cast<int32_t>(device_id));
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtResetDevice failed, ret[" << static_cast<int>(ret)
                        << "]. The details refer to 'Ascend Error Message'.";
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call aclrtResetDevice, destroy and close tsd successful, ret[" << static_cast<int>(ret) << "]";
    (void)DlogReportFinalize();
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}

bool AscendDeprecatedInterface::IsTsdOpened(const std::shared_ptr<MsContext> &ms_context_ptr) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) > 0;
}

void AscendDeprecatedInterface::AclOptimizer(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("310_multi_graph_pm");
  pm->AddPass(std::make_shared<opt::InsertPlaceholderForDynamicRNN>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
}

bool AscendDeprecatedInterface::CheckIsAscend910Soc() {
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c == nullptr) {
    return false;
  }
  std::string soc_name(soc_name_c);
  if (soc_name.find("910") == std::string::npos) {
    return false;
  }
  return true;
}

void AscendDeprecatedInterface::AclLoadModel(Buffer *om_data) {
  // check om
  MS_EXCEPTION_IF_NULL(om_data);
  ::ge::ModelHelper helper;
  ::ge::ModelData model_data;
  model_data.model_data = om_data->MutableData();
  model_data.model_len = om_data->DataSize();
  ::ge::Status ret = helper.LoadRootModel(model_data);
  if (ret != ::ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Invalid input data cannot parse to om.";
  }
}

#ifdef WITH_BACKEND
namespace {
void SetContextSocVersion(MsContext *ctx) {
  constexpr auto k910AAscendVersion = "ascend910";
  constexpr auto k910BAscendVersion = "ascend910b";
  const std::map<std::string, std::string> kAscendSocVersions = {
    {"Ascend910A", "ascend910"},    {"Ascend910B", "ascend910"},    {"Ascend910PremiumA", "ascend910"},
    {"Ascend910ProA", "ascend910"}, {"Ascend910ProB", "ascend910"}, {"Ascend910B1", "ascend910b"},
    {"Ascend910B2", "ascend910b"},  {"Ascend910B3", "ascend910b"},  {"Ascend910B4", "ascend910b"}};
  // Get default soc version.
  static std::string version;
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    version = soc_version;
  }
  auto iter = kAscendSocVersions.find(version);
  if (iter == kAscendSocVersions.end()) {
    MS_LOG(INFO) << "The soc version is not Ascend910 or ascend910b.";
    return;
  }
  if (iter->second == k910BAscendVersion) {
    ctx->set_ascend_soc_version(k910BAscendVersion);
  } else if (iter->second == k910AAscendVersion) {
    ctx->set_ascend_soc_version(k910AAscendVersion);
  }
}
}  // namespace

MSCONTEXT_REGISTER_INIT_FUNC(kAscendDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
  if (enable_ge == "1") {
    if (ctx->backend_policy() != "ge") {
      (void)ctx->set_backend_policy("ge");
    }
  } else {
    if (ctx->backend_policy() != "ms") {
      (void)ctx->set_backend_policy("ms");
    }
  }
  SetContextSocVersion(ctx);
});
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
