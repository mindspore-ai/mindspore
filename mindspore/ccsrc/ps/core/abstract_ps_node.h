/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_ABSTRACT_PS_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_ABSTRACT_PS_NODE_H_

#include "ps/core/abstract_node.h"

namespace mindspore {
namespace ps {
namespace core {
class AbstractPSNode : public AbstractNode {
 public:
  AbstractPSNode() = default;
  ~AbstractPSNode() override = default;

 protected:
  // Init the TCP connection client to the scheduler.
  bool InitClientToScheduler() override;

  // Start the heartbeat thread.
  void StartHeartbeatTimer();

 private:
  // Register collective communication initialization response methods.
  void RegisterInitCollectCommResphandler() override;

  // Register recovery response methods.
  void RegisterRecoveryRespHandler() override;

  // Indicate whether the heartbeat thread should be stopped.
  std::atomic<bool> stop_heartbeat_{false};

  // Indicate whether the heartbeat thread has been stopped.
  std::atomic<bool> heartbeat_stopped_{false};

  // Mutex the reinit tcp client to scheduler operation.
  std::mutex reinit_mutex_;

  bool DoHeartbeat();

  // Reinit the tcp connection to the scheduler if the heartbeat failed.
  bool HandleHeartbeatTimeout();
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_ABSTRACT_PS_NODE_H_
