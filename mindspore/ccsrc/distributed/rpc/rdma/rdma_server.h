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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_SERVER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_SERVER_H_

#include <string>
#include <memory>

#include "distributed/rpc/rdma/constants.h"
#include "distributed/rpc/rpc_server_base.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class BACKEND_EXPORT RDMAServer : public RPCServerBase {
 public:
  explicit RDMAServer(bool enable_ssl = false)
      : RPCServerBase(enable_ssl),
        urpc_allocator_(urpc_get_default_allocator_func()),
        dev_name_(const_cast<char *>(kDefaultIfName)),
        ip_addr_(const_cast<char *>(kDefaultIP)),
        port_(kDefaultPort) {}
  ~RDMAServer() override = default;

  bool Initialize(const std::string &url, const MemAllocateCallback &allocate_cb = {}) override;
  bool Initialize(const MemAllocateCallback &allocate_cb = {}) override;
  void Finalize() override;
  void SetMessageHandler(const MessageHandler &handler) override;

  std::string GetIP() const override;
  uint32_t GetPort() const override;

  MessageHandler message_handler_;
  struct urpc_buffer_allocator *urpc_allocator_;

 private:
  // Initialize urpc configuration according to dev_name_, ip_addr_ and port_.
  bool InitializeURPC();

  // The message callback for urpc. This method will call user-set message handler.
  static void urpc_req_handler(struct urpc_sgl *req, void *arg, struct urpc_sgl *rsp);
  // The callback after this server responding.
  static void urpc_rsp_handler(struct urpc_sgl *rsp, void *arg);

  char *dev_name_;
  char *ip_addr_;
  uint16_t port_;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RPC_RDMA_RDMA_SERVER_H_
