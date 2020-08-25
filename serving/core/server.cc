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
#include <event2/listener.h>
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

static std::pair<struct evhttp *, struct event_base *> NewHttpServer() {
  auto option_args = Options::Instance().GetArgs();
  int32_t http_port = option_args->rest_api_port;
  // init http server
  event_init();
  evthread_use_pthreads();
  struct event_base *eb = event_base_new();
  if (eb == nullptr) {
    MSI_LOG(ERROR) << "Serving Error: RESTful server start failed, new http event failed";
    std::cout << "Serving Error: RESTful server start failed, new http event failed" << std::endl;
    return std::make_pair(nullptr, nullptr);
  }
  struct evhttp *http_server = evhttp_new(eb);
  if (http_server == nullptr) {
    MSI_LOG(ERROR) << "Serving Error: RESTful server start failed, create http server faild";
    std::cout << "Serving Error: RESTful server start failed, create http server faild" << std::endl;
    event_base_free(eb);
    return std::make_pair(nullptr, nullptr);
  }

  struct sockaddr_in sin = {};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(http_port);
  auto listener =
    evconnlistener_new_bind(eb, nullptr, nullptr, LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_EXEC | LEV_OPT_CLOSE_ON_FREE, -1,
                            reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));

  if (listener == nullptr) {
    MSI_LOG_ERROR << "Serving Error: RESTful server start failed, create http listener faild, port " << http_port;
    std::cout << "Serving Error: RESTful server start failed, create http listener faild, port " << http_port
              << std::endl;
    event_base_free(eb);
    evhttp_free(http_server);
    return std::make_pair(nullptr, nullptr);
  }
  auto bound = evhttp_bind_listener(http_server, listener);
  if (bound == nullptr) {
    MSI_LOG_ERROR << "Serving Error: RESTful server start failed, bind http listener to server faild, port "
                  << http_port;
    std::cout << "Serving Error: RESTful server start failed, bind http listener  to server faild, port " << http_port
              << std::endl;
    evconnlistener_free(listener);
    event_base_free(eb);
    evhttp_free(http_server);
    return std::make_pair(nullptr, nullptr);
  }
  return std::make_pair(http_server, eb);
}

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
    MSI_LOG(ERROR) << "Serving Error: create inference session failed, device type  " << device_type << " device id "
                   << device_id;
    std::cout << "Serving Error: create inference session failed, device type  " << device_type << " device id "
              << device_id << std::endl;
    ClearEnv();
    return res;
  }
  VersionController version_controller(option_args->poll_model_wait_seconds, model_path, model_name);
  res = version_controller.Run();
  if (res != SUCCESS) {
    MSI_LOG(ERROR) << "Serving Error: load model failed, model directory " << option_args->model_path << " model name "
                   << option_args->model_name;
    std::cout << "Serving Error: load model failed, model directory " << option_args->model_path << " model name "
              << option_args->model_name << std::endl;
    ClearEnv();
    return res;
  }
  auto http_server_new_ret = NewHttpServer();
  struct evhttp *http_server = http_server_new_ret.first;
  struct event_base *eb = http_server_new_ret.second;
  if (http_server == nullptr || eb == nullptr) {
    MSI_LOG(ERROR) << "Serving Error: RESTful server start failed";
    std::cout << "Serving Error: RESTful server start failed" << std::endl;
    ClearEnv();
    return FAILED;
  }
  auto exit_http = [eb, http_server]() {
    evhttp_free(http_server);
    event_base_free(eb);
  };
  int32_t http_port = option_args->rest_api_port;
  std::string http_addr = "0.0.0.0";

  evhttp_set_timeout(http_server, 5);
  evhttp_set_gencb(http_server, http_handler_msg, nullptr);

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
    MSI_LOG(ERROR) << "Serving Error: create server failed, gRPC address " << server_address << ", RESTful address "
                   << http_addr << ":" << http_port << ", model directory " << option_args->model_path << " model name "
                   << option_args->model_name << ", device type " << option_args->device_type << ", device id "
                   << option_args->device_id;
    std::cout << "Serving Error: create server failed, gRPC address " << server_address << ", RESTful address "
              << http_addr << ":" << http_port << ", model directory " << option_args->model_path << " model name "
              << option_args->model_name << ", device type " << option_args->device_type << ", device id "
              << option_args->device_id << std::endl;
    ClearEnv();
    exit_http();
    return FAILED;
  }
  auto grpc_server_run = [&server, &server_address]() {
    MSI_LOG(INFO) << "MS Serving grpc listening on " << server_address;
    std::cout << "Serving: MS Serving gRPC start success, listening on " << server_address << std::endl;
    server->Wait();
  };
  auto http_server_run = [&eb, &http_addr, &http_port]() {
    MSI_LOG(INFO) << "MS Serving restful listening on " << http_addr << ":" << http_port;
    std::cout << "Serving: MS Serving RESTful start success, listening on " << http_addr << ":" << http_port
              << std::endl;
    event_base_dispatch(eb);
  };
  std::thread grpc_thread(grpc_server_run);
  std::thread restful_thread(http_server_run);
  auto exit_future = exit_requested.get_future();
  exit_future.wait();
  ClearEnv();
  server->Shutdown();
  event_base_loopexit(eb, nullptr);
  exit_http();
  grpc_thread.join();
  restful_thread.join();
  return SUCCESS;
}
}  // namespace serving
}  // namespace mindspore
