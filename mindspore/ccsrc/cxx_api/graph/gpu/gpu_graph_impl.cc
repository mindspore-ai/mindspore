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
#include "utils/log_adapter.h"
#include "mindspore/core/base/base_ref_utils.h"
#include "backend/session/session_factory.h"
#include "backend/session/executor_manager.h"
#include "runtime/device/kernel_runtime_manager.h"

namespace mindspore::api {
API_FACTORY_REG(GraphCell::GraphImpl, GPU, GPUGraphImpl);

GPUGraphImpl::GPUGraphImpl()
    : session_impl_(nullptr),
      graph_id_(0),
      device_id_(Context::Instance().GetDeviceID()),
      inputs_(),
      outputs_(),
      input_names_(),
      output_names_(),
      init_flag_(false),
      load_flag_(false) {}

Status GPUGraphImpl::InitEnv() {
  if (init_flag_) {
    MS_LOG(WARNING) << "Initialized again, return success.";
    return SUCCESS;
  }

  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return FAILED;
  }
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id_);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kGPUDevice);
  ms_context->set_param<bool>(MS_CTX_ENABLE_INFER_OPT, true);

  session_impl_ = session::SessionFactory::Get().Create(kGpuInferenceDevice);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << kGpuInferenceDevice
                  << " is available.";
    return FAILED;
  }

  session_impl_->Init(device_id_);
  init_flag_ = true;
  return SUCCESS;
}

Status GPUGraphImpl::FinalizeEnv() {
  if (!init_flag_) {
    MS_LOG(WARNING) << "Never initialize before, return success";
    return SUCCESS;
  }

  MS_LOG_INFO << "Start finalize env";
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();

  init_flag_ = false;
  MS_LOG(INFO) << "End finalize env";
  return SUCCESS;
}

Status GPUGraphImpl::Load() {
  // check graph type
  if (graph_->ModelType() != ModelType::kMindIR) {
    MS_LOG(ERROR) << "Unsupported model type " << graph_->ModelType();
    return INVALID_INPUTS;
  }

  const auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  auto func_graph = graph_data->GetFuncGraph();

  // init
  Status ret = InitEnv();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return FAILED;
  }

  ret = CompileGraph(func_graph);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Compile graph model failed";
    return FAILED;
  }
  session_impl_->GetModelInputsInfo(graph_id_, &inputs_, &input_names_);
  session_impl_->GetModelOutputsInfo(graph_id_, &outputs_, &output_names_);
  if (inputs_.empty() || inputs_.size() != input_names_.size()) {
    MS_LOG_ERROR << "Get model inputs info failed";
    return FAILED;
  }
  if (outputs_.empty() || outputs_.size() != output_names_.size()) {
    MS_LOG_ERROR << "Get model outputs info failed";
    return FAILED;
  }
  load_flag_ = true;
  return SUCCESS;
}

Status GPUGraphImpl::CompileGraph(const std::shared_ptr<FuncGraph> &funcGraphPtr) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    graph_id_ = session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
    return SUCCESS;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return FAILED;
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

Status GPUGraphImpl::ExecuteModel(const std::vector<Buffer> &request, std::vector<Buffer> *reply) {
  MS_EXCEPTION_IF_NULL(reply);

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    auto &item = request[i];
    auto input = inputs_[i];
    if (input->Size() != item.DataSize()) {
      MS_LOG(ERROR) << "Input " << i << " data size " << item.DataSize() << " not match model input data size "
                    << input->Size();
      return FAILED;
    }
    auto ret = memcpy_s(input->data_c(), input->Size(), item.Data(), item.DataSize());
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Tensor copy failed";
      return FAILED;
    }
    inputs.push_back(input);
  }
  vector<tensor::TensorPtr> outputs = RunGraph(inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return FAILED;
  }
  reply->clear();
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(*reply),
                 [](const tensor::TensorPtr &tensor) { return Buffer(tensor->data_c(), tensor->Size()); });
  return SUCCESS;
}

Status GPUGraphImpl::Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    Status ret = Load();
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << inputs_.size() << ", given count " << inputs.size();
    return INVALID_INPUTS;
  }

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs[i].DataSize() != inputs_[i]->Size()) {
      MS_LOG(ERROR) << "input " << i << " data size not match, required size " << inputs_[i]->Size() << ", given count "
                    << inputs[i].DataSize();
      return INVALID_INPUTS;
    }
  }
  if (ExecuteModel(inputs, outputs) != SUCCESS) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return FAILED;
  }
  if (outputs_.size() != outputs->size()) {
    MS_LOG(ERROR) << "Predict output size " << outputs->size() << " not match output size got from model info "
                  << outputs_.size();
    return FAILED;
  }

  return SUCCESS;
}

Status GPUGraphImpl::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                   std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  if (!load_flag_) {
    Status ret = Load();
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  GraphUtils::ClearIfNotNull(names);
  GraphUtils::ClearIfNotNull(shapes);
  GraphUtils::ClearIfNotNull(data_types);
  GraphUtils::ClearIfNotNull(mem_sizes);
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto &tensor = inputs_[i];
    GraphUtils::PushbackIfNotNull(names, input_names_[i]);
    GraphUtils::PushbackIfNotNull(shapes, tensor->shape());
    GraphUtils::PushbackIfNotNull(data_types, GraphUtils::TransTypeId2InferDataType(tensor->data_type()));
    GraphUtils::PushbackIfNotNull(mem_sizes, tensor->Size());
  }
  return SUCCESS;
}

Status GPUGraphImpl::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                    std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  if (!load_flag_) {
    Status ret = Load();
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "PrepareModel failed.";
      return ret;
    }
  }

  GraphUtils::ClearIfNotNull(names);
  GraphUtils::ClearIfNotNull(shapes);
  GraphUtils::ClearIfNotNull(data_types);
  GraphUtils::ClearIfNotNull(mem_sizes);
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto &tensor = outputs_[i];
    GraphUtils::PushbackIfNotNull(names, output_names_[i]);
    GraphUtils::PushbackIfNotNull(shapes, tensor->shape());
    GraphUtils::PushbackIfNotNull(data_types, GraphUtils::TransTypeId2InferDataType(tensor->data_type()));
    GraphUtils::PushbackIfNotNull(mem_sizes, tensor->Size());
  }

  return SUCCESS;
}

}  // namespace mindspore::api
