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

#ifndef MINDSPORE_CCSRC_FL_SERVER_SERVER_RECOVERY_H_
#define MINDSPORE_CCSRC_FL_SERVER_SERVER_RECOVERY_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "ps/core/recovery_base.h"
#include "ps/core/file_configuration.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "ps/ps_context.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr auto kServerRecovery = "server_recovery";

// The class helps server node to do recovery operation.
// Different from the recovery process in ps/core/node_recovery.*, this class focus on recovery of the server data. For
// example, current iteration number, learning rate, etc.
class ServerRecovery : public ps::core::RecoveryBase {
 public:
  ServerRecovery() : config_(nullptr), server_recovery_file_path_("") {}
  ~ServerRecovery() override = default;

  bool Initialize(const std::string &config_file) override;
  bool Recover() override;

  // Save server's metadata to persistent storage.
  bool Save(uint64_t current_iter);

  // If this server recovers, need to notify cluster to reach consistency.
  bool SyncAfterRecovery(const std::shared_ptr<ps::core::TcpCommunicator> &communicator, uint32_t rank_id);

 private:
  // This is the main config file set by ps context.
  std::unique_ptr<ps::core::FileConfiguration> config_;

  // The server recovery file path.
  std::string server_recovery_file_path_;

  // The server recovery file object.
  std::fstream server_recovery_file_;
  std::mutex server_recovery_file_mtx_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_SERVER_RECOVERY_H_
