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

#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "ps/core/node_info.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/file_configuration.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr int kTimeoutInSeconds = 30;
constexpr int kCommTimeoutInSeconds = 10;
constexpr int kCommTimeoutInThreeSeconds = 3;
class Node {
 public:
  Node()
      : is_ready_(false),
        is_finish_(false),
        is_already_stopped_(true),
        is_already_finished_(false),
        next_request_id_(0),
        current_cluster_state_(ClusterState::CLUSTER_STARTING) {}
  virtual ~Node() = default;

  using MessageCallback = std::function<void()>;

  virtual bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) = 0;
  virtual bool Stop() = 0;
  virtual bool Finish(const uint32_t &timeout = kTimeoutInSeconds) = 0;

  std::string node_id() const;
  uint32_t rank_id() const;
  NodeRole role() const;

  bool Wait(uint64_t request_id, const uint32_t &timeout = kCommTimeoutInSeconds);

  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &, const Protos &,
                       const void *, size_t size, const uint32_t &timeout = kCommTimeoutInSeconds);

  // Whether to enable disaster recovery.
  bool EnableRecovery() const;

 protected:
  bool WaitForStart(const uint32_t &timeout);

  // Send data synchronously
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                       const uint32_t &timeout = kCommTimeoutInSeconds);
  // Send data asynchronously
  bool SendMessageAsync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                        const Protos &protos, const void *data, size_t size);

  uint64_t AddMessageTrack(const uint32_t &expected_response);
  bool CheckMessageTrack(const uint64_t &request_id);
  void NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta);
  void set_message_callback(const uint64_t &request_id, const MessageCallback &callback);
  void ProcessSendDataResp(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                           size_t size);
  void RunMessageCallback(const uint64_t &request_id);

  NodeInfo node_info_;
  std::atomic<bool> is_ready_;
  std::atomic<bool> is_finish_;

  std::atomic<bool> is_already_stopped_;
  std::atomic<bool> is_already_finished_;
  std::atomic_uint64_t next_request_id_;

  std::mutex wait_start_mutex_;
  std::condition_variable wait_start_cond_;
  std::mutex wait_finish_mutex_;
  std::condition_variable wait_finish_cond_;
  std::mutex finish_mutex_;

  // the key is: request_id, the value is: <expected responses, actual responses>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> message_tracker_;
  std::mutex message_tracker_mutex_;
  std::condition_variable message_tracker_cond_;

  ClusterState current_cluster_state_;

  // Configuration file,The format is as follows
  //{
  // "recovery": {
  //      "storage_type": 1,
  //      "storge_file_path": "/home/cds/config.json"
  //  }
  // }
  std::unique_ptr<Configuration> config_;
  // Used to synchronize the connected nodes
  std::mutex client_mutex_;

  // the key is: request_id
  std::unordered_map<uint64_t, MessageCallback> message_callbacks_;
  std::mutex message_callbacks_mutex_;

  // the key is: request_id, the value is: <rank_id, RecvMessage>
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, VectorPtr>> receive_messages_;
  // the key is: request_id, the value is: <rank_id, RecvMessage>
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, VectorPtr>> workder_receive_messages_;
  std::map<std::pair<uint32_t, uint64_t>, bool> receive_messages_done_;
  std::mutex receive_messages_mutex_;

  // Message from the scheduler. The key is: request_id, the value is:RecvMessage.
  std::unordered_map<uint64_t, VectorPtr> received_scheduler_messages_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_H_
