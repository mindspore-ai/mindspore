/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_CLIENT_BASE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_CLIENT_BASE_H_

#include <string>
#include <memory>

#include "distributed/constants.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class BACKEND_EXPORT RPCClientBase {
 public:
  explicit RPCClientBase(bool enable_ssl) : enable_ssl_(enable_ssl) {}
  virtual ~RPCClientBase() = default;

  // Build or destroy the rpc client.
  virtual bool Initialize() { return true; }
  virtual void Finalize() {}

  // Connect to the specified server.
  // Function free_cb binds with client's each connection. It frees the real memory after message is sent to the peer.
  virtual bool Connect(
    const std::string &dst_url, size_t retry_count = 60, const MemFreeCallback &free_cb = [](void *data) {
      MS_ERROR_IF_NULL(data);
      delete static_cast<char *>(data);
      return true;
    }) {
    return true;
  }

  // Check if the connection to dst_url has been established.
  virtual bool IsConnected(const std::string &dst_url) { return false; }

  // Disconnect from the specified server.
  virtual bool Disconnect(const std::string &dst_url, size_t timeout_in_sec = 5) { return true; }

  // Send the message from the source to the destination synchronously and return the byte size by this method call.
  virtual bool SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes = nullptr) { return true; }

  // Send the message from the source to the destination asynchronously.
  virtual void SendAsync(std::unique_ptr<MessageBase> &&msg) {}

  virtual MessageBase *ReceiveSync(std::unique_ptr<MessageBase> &&msg, uint32_t timeout = 30) { return nullptr; }

  // Force the data in the send buffer to be sent out.
  virtual bool Flush(const std::string &dst_url) { return true; }

 protected:
  bool enable_ssl_;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_CLIENT_BASE_H_
