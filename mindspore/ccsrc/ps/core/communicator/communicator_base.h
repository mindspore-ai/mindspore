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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_COMMUNICATOR_BASE_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_COMMUNICATOR_BASE_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <thread>

#include "ps/core/communicator/message_handler.h"
#include "utils/log_adapter.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/node_info.h"
#include "include/backend/distributed/ps/constants.h"

namespace mindspore {
namespace ps {
namespace core {
enum class TcpUserCommand { kPush, kPull };

// CommunicatorBase is used to receive request and send response for server.
// It is the base class of HttpCommunicator and TcpCommunicator.
class CommunicatorBase {
 public:
  using MessageCallback = std::function<void(std::shared_ptr<MessageHandler>)>;
  using HttpMsgCallback = std::function<void(std::shared_ptr<HttpMessageHandler>)>;
  using OnNodeEventCallback = std::function<void(const ClusterEvent &)>;
  using TcpMsgCallback = std::function<void(std::shared_ptr<core::TcpConnection> conn,
                                            std::shared_ptr<core::MessageMeta> meta, const void *data, size_t size)>;
  CommunicatorBase() : running_(false) {}

  virtual ~CommunicatorBase();

  virtual bool Start() = 0;
  virtual bool Stop() = 0;
  // You need to call the Start() function before calling the Join() function, it will block server's main thread.
  // if you want to exit the Join() function, then you should call the Stop() function in another thread.
  void Join();

  virtual void RegisterMsgCallBack(const std::string &msg_type, const MessageCallback &cb) = 0;

  bool SendResponse(const void *rsp_data, size_t rsp_len, const std::shared_ptr<MessageHandler> &msg_handler);

  bool running() const;

 protected:
  std::unordered_map<std::string, MessageCallback> msg_callbacks_;
  std::thread running_thread_;
  bool running_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_COMMUNICATOR_BASE_H_
