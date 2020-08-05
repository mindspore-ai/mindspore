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
#include <chrono>

#include "include/infer_log.h"
#include "serving/ms_service.grpc.pb.h"
#include "core/util/option_parser.h"
#include "core/version_control/version_controller.h"
#include "core/util/file_system_operation.h"
#include "core/serving_tensor.h"

using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;

namespace mindspore {
namespace serving {

#define MSI_TIME_STAMP_START(name) auto time_start_##name = std::chrono::steady_clock::now();
#define MSI_TIME_STAMP_END(name)                                                                             \
  {                                                                                                          \
    auto time_end_##name = std::chrono::steady_clock::now();                                                 \
    auto time_cost = std::chrono::duration<double, std::milli>(time_end_##name - time_start_##name).count(); \
    MSI_LOG_INFO << #name " Time Cost # " << time_cost << " ms ---------------------";                          \
  }

Status Session::CreatDeviceSession(const std::string &device, uint32_t device_id) {
  session_ = inference::InferSession::CreateSession(device, device_id);
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "Creat Session Failed";
    return FAILED;
  }
  device_type_ = device;
  return SUCCESS;
}

Session &Session::Instance() {
  static Session instance;
  return instance;
}

Status Session::Predict(const PredictRequest &request, PredictReply &reply) {
  if (!model_loaded_) {
    MSI_LOG(ERROR) << "the model has not loaded";
    return FAILED;
  }
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "the inference session has not be initialized";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  MSI_LOG(INFO) << "run Predict";

  if (request.images_size() > 0) {
    ServingImagesRequest serving_images(request);
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    Status ret = session_->ExecuteModel(graph_id_, serving_images, serving_request, serving_reply);
    if (ret != SUCCESS) {
      MSI_LOG(ERROR) << "execute model with images return failed";
      return ret;
    }
  } else if (request.data_size() > 0) {
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    Status ret = session_->ExecuteModel(graph_id_, serving_request, serving_reply);
    if (ret != SUCCESS) {
      MSI_LOG(ERROR) << "execute model with datas return failed";
      return ret;
    }
  }

  MSI_LOG(INFO) << "run Predict finished";
  return SUCCESS;
}

Status Session::Warmup(const MindSporeModelPtr model) {
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "The CreatDeviceSession should be called, before warmup";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::string file_name = model->GetModelPath() + '/' + model->GetModelName();
  model_loaded_ = false;
  MSI_TIME_STAMP_START(LoadModelFromFile)
  auto ret = session_->LoadModelFromFile(file_name, graph_id_);
  MSI_TIME_STAMP_END(LoadModelFromFile)
  if (ret != SUCCESS) {
    MSI_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return ret;
  }
  model_loaded_ = true;
  MSI_LOG(INFO) << "Session Warmup finished";
  return SUCCESS;
}

Status Session::Clear() {
  if (session_ != nullptr) {
    session_->UnloadModel(graph_id_);
    session_->FinalizeEnv();
    session_ = nullptr;
  }
  return SUCCESS;
}

namespace {
static const uint32_t uint32max = 0x7FFFFFFF;
std::promise<void> exit_requested;

void ClearEnv() { Session::Instance().Clear(); }
void HandleSignal(int sig) { exit_requested.set_value(); }

grpc::Status CreatGRPCStatus(const Status &status) {
  switch (status.StatusCode()) {
    case SUCCESS:
      return grpc::Status::OK;
    case FAILED:
      return grpc::Status::CANCELLED;
    case INVALID_INPUTS: {
      auto status_msg = status.StatusMessage();
      if (status_msg.empty()) {
        status_msg = "The Predict Inputs do not match the Model Request!";
      }
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, status_msg);
    }
    default:
      return grpc::Status::CANCELLED;
  }
}

}  // namespace

// Service Implement
class MSServiceImpl final : public MSService::Service {
  grpc::Status Predict(grpc::ServerContext *context, const PredictRequest *request, PredictReply *reply) override {
    std::lock_guard<std::mutex> lock(mutex_);
    MSI_TIME_STAMP_START(Predict)
    auto res = Session::Instance().Predict(*request, *reply);
    MSI_TIME_STAMP_END(Predict)
    if (res != inference::SUCCESS) {
      return CreatGRPCStatus(res);
    }
    MSI_LOG(INFO) << "Finish call service Eval";
    return grpc::Status::OK;
  }

  grpc::Status Test(grpc::ServerContext *context, const PredictRequest *request, PredictReply *reply) override {
    MSI_LOG(INFO) << "TestService call";
    return grpc::Status::OK;
  }
  std::mutex mutex_;
};

Status Server::BuildAndStart() {
  // handle exit signal
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
  Status res;
  auto option_args = Options::Instance().GetArgs();
  std::string server_address = "0.0.0.0:" + std::to_string(option_args->grpc_port);
  std::string model_path = option_args->model_path;
  std::string model_name = option_args->model_name;
  std::string device_type = option_args->device_type;
  auto device_id = option_args->device_id;
  res = Session::Instance().CreatDeviceSession(device_type, device_id);
  if (res != SUCCESS) {
    MSI_LOG(ERROR) << "creat session failed";
    ClearEnv();
    return res;
  }
  VersionController version_controller(option_args->poll_model_wait_seconds, model_path, model_name);
  res = version_controller.Run();
  if (res != SUCCESS) {
    MSI_LOG(ERROR) << "load model failed";
    ClearEnv();
    return res;
  }
  MSServiceImpl ms_service;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  // Set the port is not reuseable
  auto option = grpc::MakeChannelArgumentOption(GRPC_ARG_ALLOW_REUSEPORT, 0);
  grpc::ServerBuilder serverBuilder;
  serverBuilder.SetOption(std::move(option));
  serverBuilder.SetMaxMessageSize(uint32max);
  serverBuilder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  serverBuilder.RegisterService(&ms_service);
  std::unique_ptr<grpc::Server> server(serverBuilder.BuildAndStart());
  if (server == nullptr) {
    MSI_LOG(ERROR) << "The serving server create failed";
    ClearEnv();
    return FAILED;
  }
  auto grpc_server_run = [&server]() { server->Wait(); };
  std::thread serving_thread(grpc_server_run);
  MSI_LOG(INFO) << "MS Serving listening on " << server_address;
  auto exit_future = exit_requested.get_future();
  exit_future.wait();
  ClearEnv();
  server->Shutdown();
  serving_thread.join();
  return SUCCESS;
}
}  // namespace serving
}  // namespace mindspore
