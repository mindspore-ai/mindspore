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

#include "ps/core/cluster_metadata.h"
#include "ps/core/tcp_client.h"
#include "ps/core/tcp_server.h"
#include "ps/core/abstract_node.h"

namespace mindspore {
namespace ps {
namespace core {
class ServerNode : public AbstractNode {
 public:
  ServerNode() : server_(nullptr), server_thread_(nullptr) {}
  ~ServerNode() override = default;

  bool Start(const uint32_t &timeout = ClusterMetadata::instance()->cluster_available_timeout()) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

  using RequestHandler = std::function<void(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                            DataPtr data, size_t size)>;

  void set_handler(const RequestHandler &handler);
  void Response(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta, const void *data, size_t size);

 private:
  void CreateTcpServer();
  void Initialize();
  void ProcessSendData(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta, const Protos &protos,
                       const void *data, size_t size);
  void ProcessCollectiveSendData(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                 const void *data, size_t size);

  std::shared_ptr<TcpServer> server_;
  std::unique_ptr<std::thread> server_thread_;
  RequestHandler request_handler_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_
