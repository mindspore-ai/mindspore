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
#include "core/server.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <future>

#include "mindspore/ccsrc/utils/log_adapter.h"
#include "serving/ms_service.grpc.pb.h"
#include "core/util/option_parser.h"
#include "core/version_control/version_controller.h"
#include "mindspore/ccsrc/utils/context/ms_context.h"
#include "core/util/file_system_operation.h"
#include "graphengine/third_party/fwkacllib/inc/runtime/context.h"

using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;

namespace mindspore {
namespace serving {
using MSTensorPtr = std::shared_ptr<inference::MSTensor>;

Status Session::CreatDeviceSession(const std::string &device, uint32_t device_id) {
  session_ = inference::MSSession::CreateSession(device, device_id);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Creat Session Failed";
    return FAILED;
  }
  device_type_ = device;
  return SUCCESS;
}

Session &Session::Instance() {
  static Session instance;
  return instance;
}

Status Session::Predict(const std::vector<MSTensorPtr> &inputs, inference::MultiTensor *outputs) {
  if (last_graph_ == nullptr) {
    MS_LOG(ERROR) << "the model has not loaded";
    return FAILED;
  }
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "the inference session has not be initialized";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  MS_LOG(INFO) << "run Predict";

  *outputs = session_->RunGraph(graph_id_, inputs);
  MS_LOG(INFO) << "run Predict finished";
  return SUCCESS;
}

Status Session::Warmup(const MindSporeModelPtr model) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "The CreatDeviceSession should be called, before warmup";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  size_t size = 0;
  std::string file_name = model->GetModelPath() + '/' + model->GetModelName();
  char *graphBuf = ReadFile(file_name.c_str(), &size);
  if (graphBuf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed, file name is " << file_name.c_str();
    return FAILED;
  }
  last_graph_ = inference::LoadModel(graphBuf, size, device_type_);
  if (last_graph_ == nullptr) {
    MS_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return FAILED;
  }
  graph_id_ = session_->CompileGraph(last_graph_);
  MS_LOG(INFO) << "Session Warmup finished";
  return SUCCESS;
}

Status Session::Clear() {
  session_ = nullptr;
  return SUCCESS;
}

namespace {
static const uint32_t uint32max = 0x7FFFFFFF;
std::promise<void> exit_requested;

const std::map<ms_serving::DataType, TypeId> type2id_map{
  {ms_serving::MS_UNKNOWN, TypeId::kNumberTypeBegin},   {ms_serving::MS_BOOL, TypeId::kNumberTypeBool},
  {ms_serving::MS_INT8, TypeId::kNumberTypeInt8},       {ms_serving::MS_UINT8, TypeId::kNumberTypeUInt8},
  {ms_serving::MS_INT16, TypeId::kNumberTypeInt16},     {ms_serving::MS_UINT16, TypeId::kNumberTypeUInt16},
  {ms_serving::MS_INT32, TypeId::kNumberTypeInt32},     {ms_serving::MS_UINT32, TypeId::kNumberTypeUInt32},
  {ms_serving::MS_INT64, TypeId::kNumberTypeInt64},     {ms_serving::MS_UINT64, TypeId::kNumberTypeUInt64},
  {ms_serving::MS_FLOAT16, TypeId::kNumberTypeFloat16}, {ms_serving::MS_FLOAT32, TypeId::kNumberTypeFloat32},
  {ms_serving::MS_FLOAT64, TypeId::kNumberTypeFloat64},
};

const std::map<TypeId, ms_serving::DataType> id2type_map{
  {TypeId::kNumberTypeBegin, ms_serving::MS_UNKNOWN},   {TypeId::kNumberTypeBool, ms_serving::MS_BOOL},
  {TypeId::kNumberTypeInt8, ms_serving::MS_INT8},       {TypeId::kNumberTypeUInt8, ms_serving::MS_UINT8},
  {TypeId::kNumberTypeInt16, ms_serving::MS_INT16},     {TypeId::kNumberTypeUInt16, ms_serving::MS_UINT16},
  {TypeId::kNumberTypeInt32, ms_serving::MS_INT32},     {TypeId::kNumberTypeUInt32, ms_serving::MS_UINT32},
  {TypeId::kNumberTypeInt64, ms_serving::MS_INT64},     {TypeId::kNumberTypeUInt64, ms_serving::MS_UINT64},
  {TypeId::kNumberTypeFloat16, ms_serving::MS_FLOAT16}, {TypeId::kNumberTypeFloat32, ms_serving::MS_FLOAT32},
  {TypeId::kNumberTypeFloat64, ms_serving::MS_FLOAT64},
};
const std::map<ms_serving::DataType, size_t> length_map{
  {ms_serving::MS_UNKNOWN, 0},
  {ms_serving::MS_BOOL, sizeof(bool)},
  {ms_serving::MS_INT8, sizeof(int8_t)},
  {ms_serving::MS_UINT8, sizeof(uint8_t)},
  {ms_serving::MS_INT16, sizeof(int16_t)},
  {ms_serving::MS_UINT16, sizeof(uint16_t)},
  {ms_serving::MS_INT32, sizeof(int32_t)},
  {ms_serving::MS_UINT32, sizeof(uint32_t)},
  {ms_serving::MS_INT64, sizeof(int64_t)},
  {ms_serving::MS_UINT64, sizeof(uint64_t)},
  {ms_serving::MS_FLOAT16, 2},
  {ms_serving::MS_FLOAT32, 4},
  {ms_serving::MS_FLOAT64, 8},
};
MSTensorPtr ServingTensor2MSTensor(const ms_serving::Tensor &tensor) {
  std::vector<int> shape;
  for (auto dim : tensor.tensor_shape().dims()) {
    shape.push_back(static_cast<int>(dim));
  }
  auto iter = type2id_map.find(tensor.tensor_type());
  if (iter == type2id_map.end()) {
    MS_LOG(ERROR) << "input tensor type is wrong, type is " << tensor.tensor_type();
    return nullptr;
  }
  TypeId type = iter->second;
  auto ms_tensor = std::shared_ptr<inference::MSTensor>(inference::MSTensor::CreateTensor(type, shape));
  memcpy_s(ms_tensor->MutableData(), ms_tensor->Size(), tensor.data().data(), tensor.data().size());
  return ms_tensor;
}

ms_serving::Tensor MSTensor2ServingTensor(MSTensorPtr ms_tensor) {
  ms_serving::Tensor tensor;
  ms_serving::TensorShape shape;
  for (auto dim : ms_tensor->shape()) {
    shape.add_dims(dim);
  }
  *tensor.mutable_tensor_shape() = shape;
  auto iter = id2type_map.find(ms_tensor->data_type());
  if (iter == id2type_map.end()) {
    MS_LOG(ERROR) << "input tensor type is wrong, type is " << tensor.tensor_type();
    return tensor;
  }
  tensor.set_tensor_type(iter->second);
  tensor.set_data(ms_tensor->MutableData(), ms_tensor->Size());
  return tensor;
}

void ClearEnv() {
  Session::Instance().Clear();
  inference::ExitInference();
}
void HandleSignal(int sig) { exit_requested.set_value(); }

#ifdef ENABLE_D
static rtContext_t g_ctx = nullptr;
#endif
}  // namespace

// Service Implement
class MSServiceImpl final : public MSService::Service {
  grpc::Status Predict(grpc::ServerContext *context, const PredictRequest *request, PredictReply *reply) override {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef ENABLE_D
    if (g_ctx == nullptr) {
      MS_LOG(ERROR) << "rtCtx is nullptr";
      return grpc::Status::CANCELLED;
    }
    rtError_t rt_ret = rtCtxSetCurrent(g_ctx);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "set Ascend rtCtx failed";
    }
#endif
    std::vector<MSTensorPtr> inputs;
    inference::MultiTensor outputs;
    for (int i = 0; i < request->data_size(); i++) {
      auto input = ServingTensor2MSTensor(request->data(i));
      if (input == nullptr) {
        MS_LOG(ERROR) << "Tensor convert failed";
        return grpc::Status::CANCELLED;
      }
      inputs.push_back(input);
    }
    auto res = Session::Instance().Predict(inputs, &outputs);
    if (res != SUCCESS) {
      return grpc::Status::CANCELLED;
    }
    for (const auto &tensor : outputs) {
      *reply->add_result() = MSTensor2ServingTensor(tensor);
    }
    MS_LOG(INFO) << "Finish call service Eval";
    return grpc::Status::OK;
  }

  grpc::Status Test(grpc::ServerContext *context, const PredictRequest *request, PredictReply *reply) override {
    MS_LOG(INFO) << "TestService call";
    return grpc::Status::OK;
  }
  std::mutex mutex_;
};

Status Server::BuildAndStart() {
  // handle exit signal
  signal(SIGINT, HandleSignal);
  Status res;
  auto option_args = Options::Instance().GetArgs();
  std::string server_address = "0.0.0.0:" + std::to_string(option_args->grpc_port);
  std::string model_path = option_args->model_path;
  std::string model_name = option_args->model_name;
  std::string device_type = option_args->device_type;
  auto device_id = option_args->device_id;
  res = Session::Instance().CreatDeviceSession(device_type, device_id);
  if (res != SUCCESS) {
    MS_LOG(ERROR) << "creat session failed";
    ClearEnv();
    return res;
  }
  VersionController version_controller(option_args->poll_model_wait_seconds, model_path, model_name);
  res = version_controller.Run();
  if (res != SUCCESS) {
    MS_LOG(ERROR) << "load model failed";
    ClearEnv();
    return res;
  }
#ifdef ENABLE_D
  // set d context
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE || ctx == nullptr) {
    MS_LOG(ERROR) << "the ascend device context is null";
    ClearEnv();
    return FAILED;
  }
  g_ctx = ctx;
#endif
  MSServiceImpl service;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  // Set the port is not reuseable
  auto option = grpc::MakeChannelArgumentOption(GRPC_ARG_ALLOW_REUSEPORT, 0);
  grpc::ServerBuilder builder;
  builder.SetOption(std::move(option));
  builder.SetMaxMessageSize(uint32max);
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  auto grpc_server_run = [&server]() { server->Wait(); };
  std::thread serving_thread(grpc_server_run);
  MS_LOG(INFO) << "Server listening on " << server_address << std::endl;
  auto exit_future = exit_requested.get_future();
  exit_future.wait();
  ClearEnv();
  server->Shutdown();
  serving_thread.join();
  return SUCCESS;
}
}  // namespace serving
}  // namespace mindspore
