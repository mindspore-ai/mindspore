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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_CLIENT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_CLIENT_H_

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "distributed/rpc/rdma/constants.h"
#include "distributed/rpc/rpc_client_base.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class BACKEND_EXPORT RDMAClient : public RPCClientBase {
 public:
  explicit RDMAClient(bool enable_ssl = false)
      : RPCClientBase(enable_ssl),
        dev_name_(kDefaultIfName),
        ip_addr_(kDefaultIP),
        port_(kDefaultPort),
        func_id_(0),
        urpc_allocator_(urpc_get_default_allocator_func()),
        urpc_session_(nullptr) {}
  ~RDMAClient() override = default;

  bool Initialize() override;
  void Finalize() override;
  bool Connect(
    const std::string &dst_url, size_t retry_count = 60, const MemFreeCallback &free_cb = [](void *data) {
      MS_ERROR_IF_NULL(data);
      delete static_cast<char *>(data);
      return true;
    }) override;
  bool IsConnected(const std::string &dst_url) override;
  bool Disconnect(const std::string &dst_url, size_t timeout_in_sec = 5) override;

  bool SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes = nullptr) override;
  void SendAsync(std::unique_ptr<MessageBase> &&msg) override;

  bool Flush(const std::string &dst_url) override;

  // The callback after server responding.
  static void urpc_rsp_cb(struct urpc_sgl *rsp, int err, void *arg);

 private:
  std::string dev_name_;
  std::string ip_addr_;
  uint16_t port_;
  uint32_t func_id_;

  struct urpc_buffer_allocator *urpc_allocator_;
  urpc_session_t *urpc_session_;

  // The variables for synchronization of async messages.
  std::mutex mtx_;
  std::condition_variable cv_;

  // Callback arguments when request is successfully received by peer.
  // It's used in async scenario to do releasing and synchronizing operations.
  struct req_cb_arg cb_arg_;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_CLIENT_H_
