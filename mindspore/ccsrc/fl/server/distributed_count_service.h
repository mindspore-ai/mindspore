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

#ifndef MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_
#define MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_

#include <set>
#include <string>
#include <memory>
#include <unordered_map>
#include "proto/ps.pb.h"
#include "fl/server/common.h"
#include "ps/core/server_node.h"
#include "ps/core/communicator/tcp_communicator.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr uint32_t kDefaultCountingServerRank = 0;
constexpr auto kModuleDistributedCountService = "DistributedCountService";
// The callbacks for the first count and last count event.
typedef struct {
  MessageCallback first_count_handler;
  MessageCallback last_count_handler;
} CounterHandlers;

// DistributedCountService is used for counting in the server cluster dimension. It's used for counting of rounds,
// aggregation counting, etc.

// The counting could be called by any server, but only one server has the information
// of the cluster count and we mark this server as the counting server. Other servers must communicate with this
// counting server to increase/query count number.

// On the first count or last count event, DistributedCountService on the counting server triggers the event on other
// servers by sending counter event commands. This is for the purpose of keeping server cluster's consistency.
class DistributedCountService {
 public:
  static DistributedCountService &GetInstance() {
    static DistributedCountService instance;
    return instance;
  }

  // Initialize counter service with the server node because communication is needed.
  void Initialize(const std::shared_ptr<ps::core::ServerNode> &server_node, uint32_t counting_server_rank);

  // Register message callbacks of the counting server to handle messages sent by the other servers.
  void RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator);

  // Register counter to the counting server for the name with its threshold count in server cluster dimension and
  // first/last count event callbacks.
  void RegisterCounter(const std::string &name, size_t global_threshold_count, const CounterHandlers &counter_handlers);

  // Reinitialize counter due to the change of threshold count.
  bool ReInitCounter(const std::string &name, size_t global_threshold_count);

  // Report a count to the counting server. Parameter 'id' is in case of repeated counting. Parameter 'reason' is the
  // reason why counting failed.
  bool Count(const std::string &name, const std::string &id, std::string *reason = nullptr);

  // Query whether the count reaches the threshold count for the name. If the count is the same as the threshold count,
  // this method returns true.
  bool CountReachThreshold(const std::string &name);

  // Reset the count of the name to 0.
  void ResetCounter(const std::string &name);

  // Reinitialize counting service after scaling operations are done.
  bool ReInitForScaling();

  // Returns the server rank because in some cases the callers use this rank as the 'id' for method
  // Count.
  uint32_t local_rank() { return local_rank_; }

 private:
  DistributedCountService() = default;
  ~DistributedCountService() = default;
  DistributedCountService(const DistributedCountService &) = delete;
  DistributedCountService &operator=(const DistributedCountService &) = delete;

  // Callback for the reporting count message from other servers. Only counting server will call this method.
  void HandleCountRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Callback for the querying whether threshold count is reached message from other servers. Only counting
  // server will call this method.
  void HandleCountReachThresholdRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Callback for the first/last event message from the counting server. Only other servers will call this
  // method.
  void HandleCounterEvent(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Call the callbacks when the first/last count event is triggered.
  bool TriggerCounterEvent(const std::string &name, std::string *reason = nullptr);
  bool TriggerFirstCountEvent(const std::string &name, std::string *reason = nullptr);
  bool TriggerLastCountEvent(const std::string &name, std::string *reason = nullptr);

  // Members for the communication between counting server and other servers.
  std::shared_ptr<ps::core::ServerNode> server_node_;
  std::shared_ptr<ps::core::TcpCommunicator> communicator_;
  uint32_t local_rank_;
  uint32_t server_num_;

  // Only one server will be set to do the real counting.
  uint32_t counting_server_rank_;

  // Key: name, e.g, startFLJob, updateModel, push.
  // Value: a set of id without repeatation because each work may report multiple times.
  std::unordered_map<std::string, std::set<std::string>> global_current_count_;

  // Key: name, e.g, StartFLJobCount.
  // Value: global threshold count in the server cluster dimension for this name.
  std::unordered_map<std::string, size_t> global_threshold_count_;

  // First/last count event callbacks of the name.
  std::unordered_map<std::string, CounterHandlers> counter_handlers_;

  // Because the count is increased/queried conccurently, we must ensure the operations are threadsafe.
  std::unordered_map<std::string, std::mutex> mutex_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_
