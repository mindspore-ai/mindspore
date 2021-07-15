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

#ifndef MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/abstract_node.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/communicator/communicator_base.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr char kTcpCommunicator[] = "TCP";
constexpr char kHttpCommunicator[] = "HTTP";

class ServerNode : public AbstractNode {
 public:
  ServerNode() = default;

  ~ServerNode() override = default;

  bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

  using RequestHandler =
    std::function<void(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const DataPtr &data, size_t size)>;

  void set_handler(const RequestHandler &handler);
  void Response(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta, const void *data,
                size_t size);

  std::shared_ptr<CommunicatorBase> GetOrCreateHttpComm(const std::string &ip, uint16_t port,
                                                        const std::shared_ptr<TaskExecutor> &task_executor);
  std::shared_ptr<CommunicatorBase> GetOrCreateTcpComm(const std::string &scheduler_ip, uint16_t scheduler_port,
                                                       uint32_t worker_num, uint32_t server_num,
                                                       const std::shared_ptr<TaskExecutor> &task_executor);

 private:
  void CreateTcpServer();
  void Initialize();
  void ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &protos, const void *data, size_t size);
  void ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                 const void *data, size_t size);

  RequestHandler request_handler_;
  std::unordered_map<std::string, std::shared_ptr<CommunicatorBase>> communicators_;
  std::mutex communicator_mutex_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_
