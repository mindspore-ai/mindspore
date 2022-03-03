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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_SERVER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_SERVER_H_

#include <string>
#include <memory>

#include "distributed/rpc/tcp/tcp_comm.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class TCPServer {
 public:
  TCPServer() = default;
  ~TCPServer() = default;

  // Init the tcp server using the specified url.
  bool Initialize(const std::string &url);

  // Init the tcp server using local IP and random port.
  bool Initialize();

  // Destroy the tcp server.
  void Finalize();

  // Set the message processing handler.
  void SetMessageHandler(MessageHandler handler);

  // Return the IP and port binded by this server.
  std::string GetIP();
  uint32_t GetPort();

 private:
  bool InitializeImpl(const std::string &url);

  // The basic TCP communication component used by the server.
  std::unique_ptr<TCPComm> tcp_comm_;

  std::string ip_{""};
  uint32_t port_{0};

  DISABLE_COPY_AND_ASSIGN(TCPServer);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
