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

#ifndef MINDSPORE_CCSRC_PS_CORE_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_NODE_H_

#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <utility>

#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/core/cluster_config.h"
#include "ps/core/node_info.h"
#include "ps/core/tcp_client.h"
#include "ps/core/tcp_server.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
class Node {
 public:
  Node()
      : is_ready_(false),
        is_finish_(false),
        is_timeout_(false),
        is_already_stopped_(true),
        next_request_id_(0),
        heart_beat_thread_(nullptr) {}
  virtual ~Node() = default;

  using OnNodeEventMessage = std::function<void(const NodeEvent &event)>;
  void set_callback(const OnNodeEventMessage &on_node_event_message);

  std::string node_id() const;
  uint32_t rank_id() const;

  void Wait(uint64_t request_id);

 protected:
  void Heartbeat(const std::shared_ptr<TcpClient> &client);
  void ProcessHeartbeatResp(const CommMessage &message);
  void FetchServers(const std::shared_ptr<TcpClient> &client);
  void ProcessFetchServersResp(const CommMessage &message);
  void Disconnect(const std::shared_ptr<TcpClient> &client);
  void WaitForStart();
  void WaitForDisconnect();
  void SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message);
  void SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message);
  void NotifyMessageArrival(const CommMessage &message);

  NodeInfo node_info_;
  std::atomic<bool> is_ready_;
  std::atomic<bool> is_finish_;
  std::atomic<bool> is_timeout_;
  std::atomic<bool> is_already_stopped_;
  std::atomic_uint64_t next_request_id_;
  std::unique_ptr<std::thread> heart_beat_thread_;

  OnNodeEventMessage on_node_event_message_;

  // rank_id-><ip, port>
  std::unordered_map<int, std::pair<std::string, uint16_t>> server_rank_ids_;

  // timestamp-><expected responses, actual responses>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> message_tracker_;
  std::mutex message_mutex_;
  std::condition_variable message_tracker_cond_;
  std::mutex wait_finish_mutex_;
  std::condition_variable wait_finish_cond_;
  std::mutex wait_start_mutex_;
  std::condition_variable wait_start_cond_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_H_
