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
#include <mutex>
#include <condition_variable>

#include "distributed/rpc/rpc_client_base.h"
#include "distributed/rpc/tcp/tcp_comm.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class BACKEND_EXPORT TCPClient : public RPCClientBase {
 public:
  explicit TCPClient(bool enable_ssl = false) : RPCClientBase(enable_ssl) {}
  ~TCPClient() override = default;

  // Build or destroy the TCP client.
  bool Initialize() override;
  void Finalize() override;

  // Connect to the specified server.
  // Function free_cb binds with client's each connection. It frees the real memory after message is sent to the peer.
  bool Connect(
    const std::string &dst_url, size_t retry_count = 60, const MemFreeCallback &free_cb = [](void *data) {
      MS_ERROR_IF_NULL(data);
      delete static_cast<char *>(data);
      return true;
    }) override;

  // Check if the connection to dst_url has been established.
  bool IsConnected(const std::string &dst_url) override;

  // Disconnect from the specified server.
  bool Disconnect(const std::string &dst_url, size_t timeout_in_sec = 5) override;

  // Send the message from the source to the destination synchronously and return the byte size by this method call.
  bool SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes = nullptr) override;

  // Send the message from the source to the destination asynchronously.
  void SendAsync(std::unique_ptr<MessageBase> &&msg) override;

  // Retrieve a message from tcp server specified by the input message.
  // Returns nullptr after timeout.
  MessageBase *ReceiveSync(std::unique_ptr<MessageBase> &&msg, uint32_t timeout = 30) override;

  // Force the data in the send buffer to be sent out.
  bool Flush(const std::string &dst_url) override;

 private:
  // The basic TCP communication component used by the client.
  std::unique_ptr<TCPComm> tcp_comm_{nullptr};

  // The mutex and condition variable used to synchronize the write and read of the received message returned by calling
  // the `ReceiveSync` method.
  std::mutex mutex_;
  std::condition_variable wait_msg_cond_;

  // The received message from the meta server by calling the method `ReceiveSync`.
  MessageBase *received_message_{nullptr};

  DISABLE_COPY_AND_ASSIGN(TCPClient);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
