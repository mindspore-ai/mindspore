/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/gpu/gpu_graph_impl.h"
#include <algorithm>
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "cxx_api/akg_kernel_register.h"
#include "utils/log_adapter.h"
#include "mindspore/core/base/base_ref_utils.h"
#include "backend/common/session/session_factory.h"
#include "backend/common/session/executor_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"

namespace mindspore {
API_FACTORY_REG(GraphCell::GraphImpl, GPUGraphImpl);

GPUGraphImpl::GPUGraphImpl()
    : session_impl_(nullptr),
      graph_id_(0),
      device_id_(0),
      inputs_info_(),
      outputs_info_(),
      input_names_(),
      output_names_(),
      init_flag_(false),
      load_flag_(false),
      set_device_id_flag_(false) {}

Status GPUGraphImpl::InitEnv() {
  if (init_flag_) {
    MS_LOG(WARNING) << "Initialized again, return success.";
    return kSuccess;
  }

  // Register op implemented with AKG.
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return kMCFailed;
  }
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id_);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kGPUDevice);

  // Set device id for sync data to host as cudaSetDevice is thread level config.
  bool ret = device::gpu::CudaDriver::SetDevice(UintToInt(device_id_));
  if (!ret) {
    MS_LOG(ERROR) << "Failed to set device id:" << device_id_;
    return kMCDeviceError;
  }

  auto &device_infos = graph_context_->MutableDeviceInfo();
  if (device_infos.size() != 1) {
    return kMCDeviceError;
  }
  auto gpu_info = device_infos[0]->Cast<GPUDeviceInfo>();
  if (gpu_info == nullptr) {
    return kMCDeviceError;
  }
  ms_context->set_param<bool>(MS_CTX_ENABLE_INFER_OPT, true);
  ms_context->set_param<std::string>(MS_CTX_INFER_PRECISION_MODE, gpu_info->GetPrecisionMode());

  session_impl_ = session::SessionFactory::Get().Create(kGpuInferenceDevice);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << kGpuInferenceDevice
                  << " is available.";
    return kMCFailed;
  }

  session_impl_->Init(device_id_);
  init_flag_ = true;
  return kSuccess;
}

Status GPUGraphImpl::FinalizeEnv() {
  if (!init_flag_) {
    MS_LOG(WARNING) << "Never initialize before, return success";
    return kSuccess;
  }

  MS_LOG_INFO << "Start finalize env";
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();

  init_flag_ = false;
  MS_LOG(INFO) << "End finalize env";
  return kSuccess;
}

Status GPUGraphImpl::Load(uint32_t device_id) {
  // check graph type
  if (graph_->ModelType() != ModelType::kMindIR) {
    MS_LOG(ERROR) << "Unsupported model type " << graph_->ModelType();
    return kMCInvalidInput;
  }

  const auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  auto func_graph = graph_data->GetFuncGraph();

  // init
  device_id_ = device_id;
  Status ret = InitEnv();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return kMCDeviceError;
  }

  ret = CompileGraph(func_graph);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Compile graph model failed";
    return kMCFailed;
  }
  MS_EXCEPTION_IF_NULL(session_impl_);
  session_impl_->GetModelInputsInfo(graph_id_, &inputs_info_, &input_names_);
  session_impl_->GetModelOutputsInfo(graph_id_, &outputs_info_, &output_names_);
  if (inputs_info_.empty() || inputs_info_.size() != input_names_.size()) {
    MS_LOG_ERROR << "Get model inputs info failed";
    return kMCInvalidInput;
  }
  if (outputs_info_.empty() || outputs_info_.size() != output_names_.size()) {
    MS_LOG_ERROR << "Get model outputs info failed";
    return kMCInvalidInput;
  }
  load_flag_ = true;
  return kSuccess;
}

Status GPUGraphImpl::CompileGraph(const std::shared_ptr<FuncGraph> &funcGraphPtr) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    graph_id_ = session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
    return kSuccess;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return kMCFailed;
  }
}

std::vector<tensor::TensorPtr> GPUGraphImpl::RunGraph(const std::vector<tensor::TensorPtr> &inputs) {
  try {
    VectorRef outputs;
    session_impl_->RunGraph(graph_id_, inputs, &outputs);
    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "RunGraph failed: " << e.what();
    return std::vector<tensor::TensorPtr>();
  }
}

Status GPUGraphImpl::ExecuteModel(const std::vector<MSTensor> &request, std::vector<MSTensor> *reply) {
  MS_EXCEPTION_IF_NULL(reply);

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    auto &item = request[i];
    auto input = inputs_info_[i];
    if (input->Size() != item.DataSize()) {
      MS_LOG(ERROR) << "Input " << i << " data size " << item.DataSize() << " not match model input data size "
                    << input->Size();
      return kMCInvalidInput;
    }
    auto ret = memcpy_s(input->data_c(), input->Size(), item.Data().get(), item.DataSize());
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Tensor copy failed";
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
  last_outputs_ = outputs;
  reply->clear();
  *reply = GetOutputs();
  return kSuccess;
}

Status GPUGraphImpl::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    Status ret = Load(device_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  // The `Load()` and `Run()` running in two threads. `Run()` always running in same thread.
  // It should set device id once.
  if (!set_device_id_flag_) {
    bool ret = device::gpu::CudaDriver::SetDevice(UintToInt(device_id_));
    if (!ret) {
      MS_LOG(ERROR) << "Failed to set device id:" << device_id_;
      return kMCDeviceError;
    }
    set_device_id_flag_ = true;
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
  if (ExecuteModel(inputs, outputs) != kSuccess) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return kMCFailed;
  }
  if (outputs_info_.size() != outputs->size()) {
    MS_LOG(ERROR) << "Predict output size " << outputs->size() << " not match output size got from model info "
                  << outputs_info_.size();
    return kMCFailed;
  }

  return kSuccess;
}

std::vector<MSTensor> GPUGraphImpl::GetInputs() {
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

std::vector<MSTensor> GPUGraphImpl::GetOutputs() {
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
      if (last_outputs_[i]->NeedSyncDeviceToHost()) {
        last_outputs_[i]->data_sync(false);
      }
      data = last_outputs_[i]->data_c();
      data_size = last_outputs_[i]->Size();
    }
    result[i] =
      MSTensor(output_names_[i], static_cast<enum DataType>(tensor->data_type()), tensor->shape(), data, data_size);
  }
  return result;
}

bool GPUGraphImpl::CheckDeviceSupport(mindspore::DeviceType device_type) { return device_type == kGPU; }
}  // namespace mindspore
