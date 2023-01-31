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
  explicit RDMAClient(bool enable_ssl = false) : RPCClientBase(enable_ssl) {}
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

 private:
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_CLIENT_H_
