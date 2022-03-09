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

#ifndef MINDSPORE_CCSRC_PS_CORE_PS_SERVER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_PS_SERVER_NODE_H_

#include <memory>
#include "ps/core/abstract_ps_node.h"

namespace mindspore {
namespace ps {
namespace core {
// This class is a derived class of ServerNode specialized for Parameter Server. It is used to rewrite the logic
// specific to Parameter Server mode training in ServerNode. For example, the registration of Parameter Server's Server
// node is synchronous.
class PSServerNode : public AbstractPSNode {
 public:
  PSServerNode() = default;
  ~PSServerNode() override = default;

  bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

 protected:
  // Initialize the components of parameter server node.
  void Initialize();

 private:
  // The Server node registers to the Scheduler node, and the registration of the Server node of the Parameter Server
  // is synchronous.
  void Register(const std::shared_ptr<TcpClient> &client) override;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_PS_SERVER_NODE_H_
