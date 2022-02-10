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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_CLIENT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_CLIENT_H_

#include <string>
#include <memory>

#include "distributed/rpc/tcp/tcp_comm.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class TCPClient {
 public:
  TCPClient() = default;
  ~TCPClient() = default;

  // Build or destroy the TCP client.
  bool Initialize();
  void Finalize();

  // Connect to the specified server.
  bool Connect(const std::string &dst_url, size_t timeout_in_sec = 5);

  // Disconnect from the specified server.
  bool Disconnect(const std::string &dst_url, size_t timeout_in_sec = 5);

  // Send the message from the source to the destination.
  int Send(std::unique_ptr<MessageBase> &&msg);

 private:
  // The basic TCP communication component used by the client.
  std::unique_ptr<TCPComm> tcp_comm_;

  DISABLE_COPY_AND_ASSIGN(TCPClient);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
