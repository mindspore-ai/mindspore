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

#include "include/backend/distributed/rpc/rpc_server_base.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class TCPComm;

class BACKEND_EXPORT TCPServer : public RPCServerBase {
 public:
  explicit TCPServer(bool enable_ssl = false);
  ~TCPServer() override;

  // Init the tcp server using the specified url.
  bool Initialize(const std::string &url, const MemAllocateCallback &allocate_cb = {}) override;

  // Init the tcp server using local IP and random port.
  bool Initialize(const MemAllocateCallback &allocate_cb = {}) override;

  // Destroy the tcp server.
  void Finalize() override;

  // Set the message processing handler.
  void SetMessageHandler(const MessageHandler &handler, uint32_t func_id = 0) override;

  // Return the IP and port binded by this server.
  std::string GetIP() const override;
  uint32_t GetPort() const override;

 private:
  bool InitializeImpl(const std::string &url, const MemAllocateCallback &allocate_cb);

  // The basic TCP communication component used by the server.
  std::unique_ptr<TCPComm> tcp_comm_;

  DISABLE_COPY_AND_ASSIGN(TCPServer);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
