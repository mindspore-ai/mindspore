/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/ascend/ascend_graph_impl.h"
#include <algorithm>
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "cxx_api/akg_kernel_register.h"
#include "cxx_api/utils.h"
#include "cxx_api/acl_utils.h"
#include "utils/log_adapter.h"
#include "mindspore/core/base/base_ref_utils.h"
#include "backend/common/session/executor_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/dev.h"
#include "include/common/utils/python_adapter.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context_manager.h"
#include "distributed/init.h"

namespace mindspore {
API_GRAPH_REG(kAscendDevice, AscendGraphImpl);
namespace {
constexpr auto kHcclEnable = "MS_ENABLE_HCCL";
constexpr auto kHcclGroupFile = "PARA_GROUP_FILE";

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  mindspore::python_adapter::set_python_env_flag(true);
  // init hccl from distributed
  if (!mindspore::distributed::Initialize()) {
    MS_LOG(EXCEPTION) << "InitHccl failed.";
  }
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (ms_context->backend_policy() == "ms") {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
#ifndef ENABLE_SECURITY
    runtime_instance->PreInit();
#endif
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    (void)device_context->GetDeprecatedInterface()->OpenTsd(ms_context);

    if (!runtime_instance->Init()) {
      MS_LOG(EXCEPTION) << "Runtime init failed.";
    }
  }
}

VectorRef GenerateInputsRef(const std::vector<tensor::TensorPtr> &inputs, const FuncGraphPtr &func_graph) {
  VectorRef results;
  std::size_t size = inputs.size();
  for (std::size_t i = 0; i < size; i++) {
    results.push_back(inputs[i]);
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if (results.size() != graph_params_size) {
    // Maybe some default parameter
    for (std::size_t i = results.size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      results.push_back(param_ptr->default_param());
    }
  }
  return results;
}

uint32_t GetRootGraphIdFromActorInfo(const std::string &actor_info) {
  const std::string prefix = "kernel_graph_";
  auto pos = actor_info.find(prefix);
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Cannot find prefix " << prefix << " from actor_info" << actor_info
                      << ", failed to get graph id.";
  }
  std::string first_num = "";
  for (size_t i = prefix.size(); i < actor_info.size(); ++i) {
    if (actor_info[i] >= '0' && actor_info[i] <= '9') {
      first_num.push_back(actor_info[i]);
    } else {
      break;
    }
  }
  return std::stoul(first_num);
}

void GetModelInputsInfo(const std::shared_ptr<KernelGraph> &kernel_graph, std::vector<tensor::TensorPtr> *inputs,
                        std::vector<std::string> *inputs_name) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(inputs_name);
  auto kernel_graph_inputs = kernel_graph->inputs();
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    if (!kernel_graph_inputs[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      continue;
    }
    auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
    if (!common::AnfAlgo::IsParameterWeight(parameter)) {
      auto input_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
      auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
      auto data_type = kernel_build_info->GetOutputDeviceType(0);
      auto ms_tensor = std::make_shared<tensor::Tensor>(data_type, input_shape);
      inputs->push_back(ms_tensor);
      inputs_name->push_back(parameter->name());
    }
  }
}

void GetModelOutputsInfo(const std::shared_ptr<KernelGraph> &kernel_graph, std::vector<tensor::TensorPtr> *outputs,
                         std::vector<std::string> *output_names) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_names);

  std::vector<tensor::TensorPtr> inputs;
  std::vector<std::string> input_names;
  GetModelInputsInfo(kernel_graph, &inputs, &input_names);

  VectorRef vector_outputs;
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  session::KernelMapTensor node_to_tensor;
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    vector_outputs.emplace_back(
      session::SessionBasic::CreateNodeOutputTensors(item, kernel_graph, inputs, &tensor_to_node, &node_to_tensor));
  }
  *outputs = TransformVectorRefToMultiTensor(vector_outputs);
  for (size_t i = 0; i < outputs->size(); i++) {
    output_names->push_back("output" + std::to_string(i));
  }
}
}  // namespace
AscendGraphImpl::AscendGraphImpl()
    : backend_(nullptr),
      actor_info_(""),
      kernel_graph_(),
      device_type_("Ascend"),
      device_id_(0),
      context_(nullptr),
      inputs_info_(),
      outputs_info_(),
      input_names_(),
      output_names_(),
      load_flag_(false) {}

AscendGraphImpl::~AscendGraphImpl() {}

Status AscendGraphImpl::InitEnv() {
  MS_LOG(INFO) << "Start to init env.";
  env_guard_ = MsEnvGuard::GetEnv(device_id_);
  if (env_guard_ == nullptr) {
    MS_LOG(ERROR) << "Env init failed.";
    return kMCDeviceError;
  }

  backend_ = std::make_shared<compile::MindRTBackend>(kMsConvert, kAscendDevice, device_id_);
  if (backend_ == nullptr) {
    MS_LOG(ERROR) << "DeviceContext create failed!, please make sure target device:" << kAscendDevice
                  << " is available.";
    return kMCFailed;
  }

  MS_LOG(INFO) << "InitEnv success.";
  return kSuccess;
}

Status AscendGraphImpl::CompileGraph(const std::shared_ptr<FuncGraph> &func_graph) {
  MS_ASSERT(backend_ != nullptr);
  try {
    MS_EXCEPTION_IF_NULL(func_graph);
    // perpare func graph
    auto manager = MakeManager();
    manager->AddFuncGraph(func_graph);
    func_graph->set_manager(manager);
    actor_info_ = backend_->CompileGraphs(func_graph);
    kernel_graph_ = backend_->GetGraphById(GetRootGraphIdFromActorInfo(actor_info_));
    return kSuccess;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return kMCFailed;
  }
}

std::vector<tensor::TensorPtr> AscendGraphImpl::RunGraph(const std::vector<tensor::TensorPtr> &inputs) {
  try {
    VectorRef outputs;
    backend_->RunGraph(actor_info_, GenerateInputsRef(inputs, func_graph_.lock()), &outputs);
    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "RunGraph failed: " << e.what();
    return std::vector<tensor::TensorPtr>();
  }
}

Status AscendGraphImpl::ExecuteModel(const std::vector<MSTensor> &request, std::vector<MSTensor> *reply) {
  MS_EXCEPTION_IF_NULL(reply);
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "rtCtx is nullptr";
    return kMCDeviceError;
  }
  rtError_t rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Set Ascend rtCtx failed";
    return kMCDeviceError;
  }

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    auto item = request[i];
    auto input = inputs_info_[i];
    if (input->Size() != item.DataSize()) {
      MS_LOG(ERROR) << "Input " << i << " data size " << item.DataSize() << " not match model input data size "
                    << input->Size();
      return kMCInvalidInput;
    }
    auto ret = memcpy_s(input->data_c(), input->Size(), item.MutableData(), item.DataSize());
    if (ret != EOK) {
      MS_LOG(ERROR) << "MSTensor copy failed";
      return kMCFailed;
    }
    inputs.push_back(input);
  }
  last_inputs_ = inputs;
  std::vector<tensor::TensorPtr> outputs = RunGraph(inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return kMCFailed;
  }
  for (const auto &out : outputs) {
    out->data_sync();
  }
  last_outputs_ = outputs;
  reply->clear();
  *reply = GetOutputs();
  return kSuccess;
}

std::vector<MSTensor> AscendGraphImpl::GetInputs() {
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return {};
    }
  }

  std::vector<MSTensor> result(inputs_info_.size());
  for (size_t i = 0; i < inputs_info_.size(); ++i) {
    auto &tensor = inputs_info_[i];
    void *data = nullptr;
    size_t data_size = tensor->Size();
    if (i < last_inputs_.size()) {
      data = last_inputs_[i]->data_c();
      data_size = last_inputs_[i]->Size();
    }
    result[i] =
      MSTensor(input_names_[i], static_cast<enum DataType>(tensor->data_type()), tensor->shape(), data, data_size);
  }
  return result;
}

std::vector<MSTensor> AscendGraphImpl::GetOutputs() {
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return {};
    }
  }

  std::vector<MSTensor> result(outputs_info_.size());
  for (size_t i = 0; i < outputs_info_.size(); ++i) {
    auto &tensor = outputs_info_[i];
    void *data = nullptr;
    size_t data_size = tensor->Size();
    if (i < last_outputs_.size()) {
      data = last_outputs_[i]->data_c();
      data_size = last_outputs_[i]->Size();
    }
    result[i] =
      MSTensor(output_names_[i], static_cast<enum DataType>(tensor->data_type()), tensor->shape(), data, data_size);
  }
  return result;
}

Status AscendGraphImpl::Load(uint32_t device_id) {
  // check graph type
  if (graph_->ModelType() != ModelType::kMindIR) {
    MS_LOG(ERROR) << "Unsupported model type " << graph_->ModelType();
    return kMCInvalidInput;
  }

  const auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  auto func_graph = graph_data->GetFuncGraph();
  func_graph_ = func_graph;

  // init
  device_id_ = device_id;
  Status ret = InitEnv();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return ret;
  }

  // load model
  if (!load_flag_) {
    ret = CompileGraph(func_graph);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Compile graph model failed";
      return ret;
    }
    auto kg = kernel_graph_.lock();
    MS_EXCEPTION_IF_NULL(backend_);
    MS_EXCEPTION_IF_NULL(kg);
    GetModelInputsInfo(kg, &inputs_info_, &input_names_);
    GetModelOutputsInfo(kg, &outputs_info_, &output_names_);
    if (inputs_info_.size() != input_names_.size()) {
      MS_LOG_ERROR << "Get model inputs info failed";
      return kMCInvalidInput;
    }
    if (outputs_info_.size() != output_names_.size()) {
      MS_LOG_ERROR << "Get model outputs info failed";
      return kMCInvalidInput;
    }

    // save d context
    rtError_t rt_ret = rtCtxGetCurrent(&context_);
    if (rt_ret != RT_ERROR_NONE || context_ == nullptr) {
      MS_LOG(ERROR) << "the ascend device context is null";
      return kMCDeviceError;
    }

    MS_LOG(INFO) << "Load model success";
    load_flag_ = true;
  }

  rtError_t rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed";
    return kMCDeviceError;
  }

  return kSuccess;
}

Status AscendGraphImpl::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  if (inputs.size() != inputs_info_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << inputs_info_.size() << ", given count "
                  << inputs.size();
    return kMCInvalidInput;
  }

  for (size_t i = 0; i < inputs_info_.size(); ++i) {
    if (inputs[i].DataSize() != inputs_info_[i]->Size()) {
      MS_LOG(ERROR) << "input " << i << " data size not match, required size " << inputs_info_[i]->Size()
                    << ", given count " << inputs[i].DataSize();
      return kMCInvalidInput;
    }
  }

  Status ret = ExecuteModel(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return ret;
  }
  if (outputs_info_.size() != outputs->size()) {
    MS_LOG(ERROR) << "Predict output size " << outputs->size() << " not match output size got from model info "
                  << outputs_info_.size();
    return kMCFailed;
  }

  return kSuccess;
}

AscendGraphImpl::MsEnvGuard::MsEnvGuard(uint32_t device_id) : device_id_(device_id) {
  MS_LOG(INFO) << "Start to init device " << device_id;
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    errno_ = kMCFailed;
    return;
  }

  auto env_hccl_mode = common::GetEnv(kHcclEnable);
  if (!env_hccl_mode.empty() && env_hccl_mode != std::to_string(0)) {
    MS_LOG(INFO) << "Enable hccl parallel mode.";
    ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  }

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id_);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);

  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    InitHccl();
    auto para_group_file = common::GetEnv(kHcclGroupFile);
    if (para_group_file.empty()) {
      MS_LOG(INFO) << "Cannot get Env " << kHcclGroupFile << ", skip.";
    } else {
      MS_LOG(INFO) << "Get env " << kHcclGroupFile << " success: " << para_group_file;
      if (!CreateGroupsByCkptFile(para_group_file)) {
        MS_LOG(ERROR) << "CreateGroupsByCkptFile failed.";
        errno_ = kMCFailed;
        return;
      }
    }
  } else {
    auto ret = rtSetDevice(static_cast<int32_t>(device_id_));
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id_ << " call rtSetDevice failed, ret[" << static_cast<int>(ret) << "]";
    }
  }

  MS_LOG(INFO) << "Device " << device_id << " init env success.";
  errno_ = kSuccess;
}

AscendGraphImpl::MsEnvGuard::~MsEnvGuard() {
  MS_LOG(INFO) << "Start finalize device " << device_id_;
  try {
    session::ExecutorManager::Instance().Clear();
    device::KernelRuntimeManager::Instance().ClearRuntimeResource();

    auto ms_context = MsContext::GetInstance();
    if (ms_context == nullptr) {
      MS_LOG(ERROR) << "Get Context failed!";
      return;
    }

    if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      PythonEnvGuard guard;
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
      if (!device_context->GetDeprecatedInterface()->CloseTsd(ms_context, false)) {
        MS_LOG(ERROR) << "CloseTsd failed!";
        return;
      }
    } else {
      auto ret = rtDeviceReset(static_cast<int32_t>(device_id_));
      if (ret != RT_ERROR_NONE) {
        MS_LOG(ERROR) << "Device " << device_id_ << " call rtDeviceReset failed, ret[" << static_cast<int>(ret) << "]";
        return;
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendGraphImpl MsEnvGuard destructor run failed, error message : " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "AscendGraphImpl MsEnvGuard destructor run failed, unknown error occurred.";
  }
  MS_LOG(INFO) << "End finalize device " << device_id_;
}

std::shared_ptr<AscendGraphImpl::MsEnvGuard> AscendGraphImpl::MsEnvGuard::GetEnv(uint32_t device_id) {
  std::shared_ptr<MsEnvGuard> acl_env;
  std::lock_guard<std::mutex> lock(global_ms_env_mutex_);
  auto iter = global_ms_env_.find(device_id);
  if (iter != global_ms_env_.end()) {
    acl_env = iter->second.lock();
  }

  if (acl_env != nullptr) {
    MS_LOG(INFO) << "Env has been initialized, skip.";
    return acl_env;
  }

  acl_env = std::make_shared<MsEnvGuard>(device_id);
  if (acl_env->GetErrno() != kSuccess) {
    MS_LOG(ERROR) << "Init ascend env Failed";
    return nullptr;
  }

  global_ms_env_.emplace(device_id, acl_env);
  MS_LOG(INFO) << "Env init success";
  return acl_env;
}

bool AscendGraphImpl::CheckDeviceSupport(mindspore::DeviceType device_type) {
  // for Ascend, only support kAscend and kAscend910
  if (device_type != kAscend && device_type != kAscend910) {
    return false;
  }
  return IsAscend910Soc();
}

std::map<uint32_t, std::weak_ptr<AscendGraphImpl::MsEnvGuard>> AscendGraphImpl::MsEnvGuard::global_ms_env_;
std::mutex AscendGraphImpl::MsEnvGuard::global_ms_env_mutex_;

PythonEnvGuard::PythonEnvGuard() : origin_init_status_(PythonIsInited()) { InitPython(); }

PythonEnvGuard::~PythonEnvGuard() {
  // finalize when init by this
  try {
    if (!origin_init_status_) {
      FinalizePython();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "PythonEnvGuard destructor run failed, error message : " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "PythonEnvGuard destructor run failed, unknown error occurred.";
  }
}

bool PythonEnvGuard::PythonIsInited() const { return Py_IsInitialized() != 0; }

void PythonEnvGuard::InitPython() const {
  if (!PythonIsInited()) {
    Py_Initialize();
  }
}

void PythonEnvGuard::FinalizePython() const {
  if (PythonIsInited()) {
    Py_Finalize();
  }
}
}  // namespace mindspore
