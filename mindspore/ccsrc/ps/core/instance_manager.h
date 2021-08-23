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

#ifndef MINDSPORE_CCSRC_PS_CORE_INSTANCE_MANAGER_H_
#define MINDSPORE_CCSRC_PS_CORE_INSTANCE_MANAGER_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ps/core/communicator/tcp_client.h"
#include "ps/core/node_manager.h"
#include "ps/core/node.h"
#include "ps/core/communicator/request_process_result_code.h"
#include "ps/constants.h"
#include "utils/log_adapter.h"
#include "ps/core/communicator/communicator_base.h"

namespace mindspore {
namespace ps {
namespace core {
// The class helps scheduler node to do new instance or query instance operation for the cluster.
class InstanceManager {
 public:
  explicit InstanceManager(Node *const node) : node_(node) {}
  ~InstanceManager() = default;

  // When the scheduler receives the new instance message, it will send this message to the workers and servers.
  void NewInstanceAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager, const std::string &body,
                        const uint64_t &request_id, const NodeInfo &node_info);
  // When the scheduler receives the query instance message, it will send this message to the server0.
  void QueryInstanceAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager,
                          const uint64_t &request_id, const NodeInfo &node_info);
  // When the scheduler receives the enable instance message, it will send this message to the server0.
  void EnableFLSAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager, const uint64_t &request_id,
                      const NodeInfo &node_info);
  // When the scheduler receives the disable instance message, it will send this message to the server0.
  void DisableFLSAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager, const uint64_t &request_id,
                       const NodeInfo &node_info);

 private:
  // The node_ will only be instantiated with scheduler node.
  Node *const node_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_INSTANCE_MANAGER_H_
