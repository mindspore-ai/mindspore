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

#include "distributed/rpc/rdma/rdma_client.h"

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAClient::Initialize() { return true; }

void RDMAClient::Finalize() {}

bool RDMAClient::Connect(const std::string &dst_url, size_t retry_count, const MemFreeCallback &free_cb) {
  return true;
}

bool RDMAClient::IsConnected(const std::string &dst_url) { return false; }

bool RDMAClient::Disconnect(const std::string &dst_url, size_t timeout_in_sec) { return true; }

bool RDMAClient::SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes) { return true; }

void RDMAClient::SendAsync(std::unique_ptr<MessageBase> &&msg) {}

bool RDMAClient::Flush(const std::string &dst_url) { return true; }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
