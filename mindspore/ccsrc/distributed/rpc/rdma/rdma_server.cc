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

#include "distributed/rpc/rdma/rdma_server.h"

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAServer::Initialize(const std::string &url, const MemAllocateCallback &allocate_cb) { return true; }

void RDMAServer::Finalize() {}

void RDMAServer::SetMessageHandler(const MessageHandler &handler) {}

std::string RDMAServer::GetIP() const { return ""; }

uint32_t RDMAServer::GetPort() const { return 0; }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
