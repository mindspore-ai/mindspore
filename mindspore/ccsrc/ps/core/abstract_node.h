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

#ifndef MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_

#include <utility>
#include <string>
#include <memory>

#include "ps/core/node.h"

namespace mindspore {
namespace ps {
namespace core {
class AbstractNode : public Node {
 public:
  AbstractNode() : heart_beat_thread_(nullptr), client_to_scheduler_thread_(nullptr), client_to_scheduler_(nullptr) {}
  ~AbstractNode() override = default;

  bool BroadcastToServers(const std::string &message, const uint32_t &timeout = kCommTimeoutInSeconds);
  void set_event_callback(const OnNodeEventMessage &on_node_event_message);

 protected:
  void Register(const std::shared_ptr<TcpClient> &client);
  void ProcessRegisterResp(const CommMessage &message);
  void Heartbeat(const std::shared_ptr<TcpClient> &client);
  void ProcessHeartbeatResp(const CommMessage &message);
  void FetchServers(const std::shared_ptr<TcpClient> &client);
  void ProcessFetchServersResp(const CommMessage &message);
  bool Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout);
  bool WaitForDisconnect(const uint32_t &timeout);
  bool InitClientToScheduler();

  std::unique_ptr<std::thread> heart_beat_thread_;
  std::unique_ptr<std::thread> client_to_scheduler_thread_;
  std::shared_ptr<TcpClient> client_to_scheduler_;
  OnNodeEventMessage on_node_event_message_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
