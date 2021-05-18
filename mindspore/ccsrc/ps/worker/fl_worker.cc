/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "ps/worker/fl_worker.h"

namespace mindspore {
namespace ps {
namespace worker {
void FLWorker::Run() {
  worker_num_ = PSContext::instance()->worker_num();
  server_num_ = PSContext::instance()->server_num();
  scheduler_ip_ = PSContext::instance()->scheduler_ip();
  scheduler_port_ = PSContext::instance()->scheduler_port();
  PSContext::instance()->cluster_config().scheduler_host = scheduler_ip_;
  PSContext::instance()->cluster_config().scheduler_port = scheduler_port_;
  PSContext::instance()->cluster_config().initial_worker_num = worker_num_;
  PSContext::instance()->cluster_config().initial_server_num = server_num_;
  MS_LOG(INFO) << "Initialize cluster config for worker. Worker number:" << worker_num_
               << ", Server number:" << server_num_ << ", Scheduler ip:" << scheduler_ip_
               << ", Scheduler port:" << scheduler_port_;
  worker_node_ = std::make_shared<core::WorkerNode>();
  worker_node_->Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  return;
}

bool FLWorker::SendToServer(uint32_t server_rank, void *data, size_t size, core::TcpUserCommand command,
                            std::shared_ptr<std::vector<unsigned char>> *output) {
  std::shared_ptr<unsigned char[]> message;
  std::unique_ptr<unsigned char[]> message_addr = std::make_unique<unsigned char[]>(size);
  MS_EXCEPTION_IF_NULL(message_addr);
  message = std::move(message_addr);
  MS_EXCEPTION_IF_NULL(message);

  uint64_t src_size = size;
  uint64_t dst_size = size;
  int ret = memcpy_s(message.get(), dst_size, data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return false;
  }

  if (output != nullptr) {
    if (!worker_node_->Send(core::NodeRole::SERVER, server_rank, message, size, static_cast<int>(command), output)) {
      MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
      return false;
    }
  } else {
    if (!worker_node_->Send(core::NodeRole::SERVER, server_rank, message, size, static_cast<int>(command))) {
      MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
      return false;
    }
  }
  return true;
}
}  // namespace worker
}  // namespace ps
}  // namespace mindspore
