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
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <utility>
#include <tuple>
#include <map>

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
constexpr int kTimeoutInSeconds = 30;
constexpr int kCommTimeoutInSeconds = 3;
class Node {
 public:
  Node()
      : is_ready_(false),
        is_finish_(false),
        is_timeout_(false),
        is_already_stopped_(true),
        is_already_finished_(false),
        next_request_id_(0) {}
  virtual ~Node() = default;

  using OnNodeEventMessage = std::function<void(const NodeEvent &event)>;
  using MessageCallback = std::function<void()>;

  virtual bool Start(const uint32_t &timeout = kTimeoutInSeconds) = 0;
  virtual bool Stop() = 0;
  virtual bool Finish(const uint32_t &timeout = kTimeoutInSeconds) = 0;

  std::string node_id() const;
  uint32_t rank_id() const;
  NodeRole role() const;

  bool Wait(uint64_t request_id, const uint32_t &timeout = kCommTimeoutInSeconds);

  virtual bool Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                    const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                    const std::vector<std::string> &data, const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                    CommMessage *comm_message_resp, const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                    const std::vector<std::string> &data, std::vector<CommMessage *> *comm_message_resp,
                    const uint32_t &timeout = kCommTimeoutInSeconds);

 protected:
  bool WaitForStart(const uint32_t &timeout);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                       const uint32_t &timeout = kCommTimeoutInSeconds);
  void SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message);
  const std::shared_ptr<TcpClient> &GetOrCreateTcpClient(const int &rank_id);
  void ProcessSendDataResp(const CommMessage &message);
  void RunMessageCallback(const uint64_t &request_id);
  void set_message_callback(const uint64_t &request_id, const MessageCallback &message_callback);
  void NotifyMessageArrival(const CommMessage &message);

  NodeInfo node_info_;
  std::atomic<bool> is_ready_;
  std::atomic<bool> is_finish_;
  std::atomic<bool> is_timeout_;
  std::atomic<bool> is_already_stopped_;
  std::atomic<bool> is_already_finished_;
  std::atomic_uint64_t next_request_id_;

  // <NodeRole,rank_id>-><ip, port>
  std::map<std::pair<NodeRole, uint32_t>, std::pair<std::string, uint16_t>> nodes_address_;
  // rank_id->tcpclient
  std::unordered_map<int, std::shared_ptr<TcpClient>> connected_nodes_;

  // request_id-><expected responses, actual responses>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> message_tracker_;
  std::mutex message_tracker_mutex_;
  std::condition_variable message_tracker_cond_;
  std::mutex wait_finish_mutex_;
  std::condition_variable wait_finish_cond_;
  std::mutex wait_start_mutex_;
  std::condition_variable wait_start_cond_;
  std::mutex finish_mutex_;
  std::mutex client_mutex_;

  // request_id -> <rank_id, CommMessage>
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, CommMessage>> receive_messages_;
  std::mutex receive_messages_mutex_;
  // request_id -> MessageCallback
  std::unordered_map<uint64_t, MessageCallback> message_callbacks_;
  std::mutex message_callbacks_mutex_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_H_
