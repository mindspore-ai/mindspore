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
#include <evhttp.h>
#include <event.h>
#include <event2/thread.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "include/infer_log.h"
#include "serving/ms_service.grpc.pb.h"
#include "core/util/option_parser.h"
#include "core/version_control/version_controller.h"
#include "core/session.h"
#include "core/serving_tensor.h"
#include "core/http_process.h"


using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;

namespace mindspore {
namespace serving {

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

  // init http server
  struct evhttp *http_server = NULL;
  struct event_base *eb = NULL;
  int32_t http_port = option_args->rest_api_port;
  std::string http_addr = "0.0.0.0";
  event_init();
  evthread_use_pthreads();
  eb = event_base_new();
  http_server = evhttp_new(eb);
  evhttp_bind_socket_with_handle(http_server, http_addr.c_str(), http_port);
  //  http_server = evhttp_start(http_addr.c_str(), http_port);
  if (http_server == NULL) {
    MSI_LOG(ERROR) << "http server start failed.";
    return res;
  }
  evhttp_set_timeout(http_server, 5);
  evhttp_set_gencb(http_server, http_handler_msg, NULL);

  // grpc server
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
  auto grpc_server_run = [&server, &server_address]() {
    MSI_LOG(INFO) << "MS Serving grpc listening on " << server_address;
    server->Wait();
  };
  auto http_server_run = [&eb, &http_addr, &http_port]() {
    MSI_LOG(INFO) << "MS Serving restful listening on " << http_addr << ":" << http_port;
    event_base_dispatch(eb);
  };
  std::thread grpc_thread(grpc_server_run);
  std::thread restful_thread(http_server_run);
  auto exit_future = exit_requested.get_future();
  exit_future.wait();
  ClearEnv();
  server->Shutdown();
  event_base_loopexit(eb, NULL);
  grpc_thread.join();
  restful_thread.join();
  return SUCCESS;
}
}  // namespace serving
}  // namespace mindspore
