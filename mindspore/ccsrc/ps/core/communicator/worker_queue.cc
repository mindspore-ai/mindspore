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

#include "ps/core/communicator/worker_queue.h"

namespace mindspore {
namespace ps {
namespace core {
bool WorkerQueue::Initialize(int fd, std::unordered_map<std::string, OnRequestReceive *> handlers) {
  evbase_ = event_base_new();
  MS_EXCEPTION_IF_NULL(evbase_);
  struct evhttp *http = evhttp_new(evbase_);
  MS_EXCEPTION_IF_NULL(http);
  int result = evhttp_accept_socket(http, fd);
  if (result < 0) {
    MS_LOG(ERROR) << "Evhttp accept socket failed!";
    return false;
  }

  for (const auto &handler : handlers) {
    auto TransFunc = [](struct evhttp_request *req, void *arg) {
      MS_EXCEPTION_IF_NULL(req);
      MS_EXCEPTION_IF_NULL(arg);
      auto httpReq = std::make_shared<HttpMessageHandler>();
      httpReq->set_request(req);
      httpReq->InitHttpMessage();
      OnRequestReceive *func = reinterpret_cast<OnRequestReceive *>(arg);
      (*func)(httpReq);
    };

    // O SUCCESS,-1 ALREADY_EXIST,-2 FAILURE
    int ret = evhttp_set_cb(http, handler.first.c_str(), TransFunc, reinterpret_cast<void *>(handler.second));
    std::string log_prefix = "Ev http register handle of:";
    if (ret == 0) {
      MS_LOG(INFO) << log_prefix << handler.first.c_str() << " success.";
    } else if (ret == -1) {
      MS_LOG(WARNING) << log_prefix << handler.first.c_str() << " exist.";
    } else {
      MS_LOG(ERROR) << log_prefix << handler.first.c_str() << " failed.";
      return false;
    }
  }
  return true;
}

void WorkerQueue::Run() {
  MS_LOG(INFO) << "Start http server!";
  MS_EXCEPTION_IF_NULL(evbase_);
  int ret = event_base_dispatch(evbase_);
  if (ret == 0) {
    MS_LOG(INFO) << "Event base dispatch success!";
  } else if (ret == 1) {
    MS_LOG(ERROR) << "Event base dispatch failed with no events pending or active!";
  } else if (ret == -1) {
    MS_LOG(ERROR) << "Event base dispatch failed with error occurred!";
  } else {
    MS_LOG(ERROR) << "Event base dispatch with unexpected error code!";
  }

  if (evbase_) {
    event_base_free(evbase_);
    evbase_ = nullptr;
  }
}

void WorkerQueue::Stop() {
  MS_LOG(INFO) << "Stop http server!";

  int ret = event_base_loopbreak(evbase_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "event base loop break failed!";
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
