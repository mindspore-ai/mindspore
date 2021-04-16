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
#include "minddata/dataset/engine/gnn/graph_data_server.h"

#include <algorithm>
#include <functional>
#include <iterator>

#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
namespace gnn {

GraphDataServer::GraphDataServer(const std::string &dataset_file, int32_t num_workers, const std::string &hostname,
                                 int32_t port, int32_t client_num, bool auto_shutdown)
    : dataset_file_(dataset_file),
      num_workers_(num_workers),
      client_num_(client_num),
      max_connected_client_num_(0),
      auto_shutdown_(auto_shutdown),
      state_(kGdsUninit) {
  tg_ = std::make_unique<TaskGroup>();
  graph_data_impl_ = std::make_unique<GraphDataImpl>(dataset_file, num_workers, true);
#if !defined(_WIN32) && !defined(_WIN64)
  service_impl_ = std::make_unique<GraphDataServiceImpl>(this, graph_data_impl_.get());
  async_server_ = std::make_unique<GraphDataGrpcServer>(hostname, port, service_impl_.get());
#endif
}

Status GraphDataServer::Init() {
#if defined(_WIN32) || defined(_WIN64)
  RETURN_STATUS_UNEXPECTED("Graph data server is not supported in Windows OS");
#else
  set_state(kGdsInitializing);
  RETURN_IF_NOT_OK(async_server_->Run());
  RETURN_IF_NOT_OK(tg_->CreateAsyncTask("init graph data impl", std::bind(&GraphDataServer::InitGraphDataImpl, this)));
  for (int32_t i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(
      tg_->CreateAsyncTask("start async rpc service", std::bind(&GraphDataServer::StartAsyncRpcService, this)));
  }
  if (auto_shutdown_) {
    RETURN_IF_NOT_OK(
      tg_->CreateAsyncTask("judge auto shutdown server", std::bind(&GraphDataServer::JudgeAutoShutdownServer, this)));
  }
  return Status::OK();
#endif
}

Status GraphDataServer::InitGraphDataImpl() {
  TaskManager::FindMe()->Post();
  Status s = graph_data_impl_->Init();
  if (s.IsOk()) {
    set_state(kGdsRunning);
  } else {
    (void)Stop();
  }
  return s;
}

#if !defined(_WIN32) && !defined(_WIN64)
Status GraphDataServer::StartAsyncRpcService() {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(async_server_->HandleRequest());
  return Status::OK();
}
#endif

Status GraphDataServer::JudgeAutoShutdownServer() {
  TaskManager::FindMe()->Post();
  while (true) {
    if (auto_shutdown_ && (max_connected_client_num_ >= client_num_) && (client_pid_.size() == 0)) {
      MS_LOG(INFO) << "All clients have been unregister, automatically exit the server.";
      RETURN_IF_NOT_OK(Stop());
      break;
    }
    if (state_ == kGdsStopped) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  return Status::OK();
}

Status GraphDataServer::Stop() {
#if !defined(_WIN32) && !defined(_WIN64)
  async_server_->Stop();
#endif
  set_state(kGdsStopped);
  graph_data_impl_.reset();
  return Status::OK();
}

Status GraphDataServer::ClientRegister(int32_t pid) {
  std::unique_lock<std::mutex> lck(mutex_);
  MS_LOG(INFO) << "client register pid:" << std::to_string(pid);
  client_pid_.emplace(pid);
  if (client_pid_.size() > max_connected_client_num_) {
    max_connected_client_num_ = client_pid_.size();
  }
  return Status::OK();
}
Status GraphDataServer::ClientUnRegister(int32_t pid) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto itr = client_pid_.find(pid);
  if (itr != client_pid_.end()) {
    client_pid_.erase(itr);
    MS_LOG(INFO) << "client unregister pid:" << std::to_string(pid);
  }
  return Status::OK();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
