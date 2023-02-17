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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_SERVER_BASE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_SERVER_BASE_H_

#include <string>
#include <memory>

#include "distributed/constants.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class BACKEND_EXPORT RPCServerBase {
 public:
  explicit RPCServerBase(bool enable_ssl) : ip_(""), port_(0), enable_ssl_(enable_ssl) {}
  virtual ~RPCServerBase() = default;

  // Init server using the specified url, with memory allocating function.
  virtual bool Initialize(const std::string &url, const MemAllocateCallback &allocate_cb = {}) { return true; }

  // Init server using local IP and random port.
  virtual bool Initialize(const MemAllocateCallback &allocate_cb = {}) { return true; }

  // Destroy the tcp server.
  virtual void Finalize() {}

  // Set the message processing handler.
  virtual void SetMessageHandler(const MessageHandler &handler) {}

  // Return the IP and port bound to this server.
  virtual std::string GetIP() const { return ip_; }
  virtual uint32_t GetPort() const { return port_; }

 protected:
  std::string ip_;
  uint32_t port_;

  bool enable_ssl_;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RPC_RPC_SERVER_BASE_H_
