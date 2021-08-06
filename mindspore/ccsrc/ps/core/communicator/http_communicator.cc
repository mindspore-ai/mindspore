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

#include "ps/core/communicator/http_communicator.h"
#include <memory>
#include "common/thread_pool.h"

namespace mindspore {
namespace ps {
namespace core {
bool HttpCommunicator::Start() {
  MS_LOG(INFO) << "Initialize http server IP:" << ip_ << ", PORT:" << port_;
  if (!http_server_->InitServer()) {
    MS_LOG(EXCEPTION) << "The communicator init http server failed.";
  }
  MS_EXCEPTION_IF_NULL(http_server_);
  if (!http_server_->Start()) {
    MS_LOG(EXCEPTION) << "Http server starting failed.";
  }
  MS_LOG(INFO) << "Http communicator started.";

  running_thread_ = std::thread([&]() {
    while (running_) {
      std::this_thread::yield();
    }
  });
  return true;
}

bool HttpCommunicator::Stop() {
  MS_EXCEPTION_IF_NULL(http_server_);
  if (!http_server_->Stop()) {
    MS_LOG(ERROR) << "Stopping http server failed.";
    return false;
  }
  running_ = false;
  return true;
}

void HttpCommunicator::RegisterMsgCallBack(const std::string &msg_type, const MessageCallback &cb) {
  msg_callbacks_[msg_type] = cb;
  http_msg_callbacks_[msg_type] = std::bind(
    [&](std::shared_ptr<HttpMessageHandler> http_msg) -> void {
      std::shared_ptr<MessageHandler> http_msg_handler = std::make_shared<HttpMsgHandler>(http_msg);
      MS_EXCEPTION_IF_NULL(http_msg_handler);
      msg_callbacks_[msg_type](http_msg_handler);
      return;
    },
    std::placeholders::_1);

  std::string url = "/";
  url += msg_type;
  MS_EXCEPTION_IF_NULL(http_server_);
  bool is_succeed = http_server_->RegisterRoute(url, &http_msg_callbacks_[msg_type]);
  if (!is_succeed) {
    MS_LOG(EXCEPTION) << "Http server register handler for url " << url << " failed.";
  }
  return;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
