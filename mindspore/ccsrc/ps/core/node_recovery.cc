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

#include "ps/core/node_recovery.h"

namespace mindspore {
namespace ps {
namespace core {
bool NodeRecovery::Recover() {
  if (recovery_storage_ == nullptr) {
    return false;
  }
  // 1. recover worker num
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  if (recovery_storage_->Exists(kRecoveryWorkerNum)) {
    int32_t worker_num = std::strtol(recovery_storage_->Get(kRecoveryWorkerNum, "").c_str(), nullptr, kBase);
    node_->set_worker_num(worker_num);
  } else {
    node_->set_worker_num(PSContext::instance()->cluster_config().initial_worker_num);
  }

  // 2. recover server num
  if (recovery_storage_->Exists(kRecoveryServerNum)) {
    int32_t server_num = std::strtol(recovery_storage_->Get(kRecoveryServerNum, "").c_str(), nullptr, kBase);
    node_->set_server_num(server_num);
  } else {
    node_->set_server_num(PSContext::instance()->cluster_config().initial_server_num);
  }

  // 3. recover scheduler ip
  if (recovery_storage_->Exists(kRecoverySchedulerIp)) {
    std::string scheduler_ip = recovery_storage_->Get(kRecoverySchedulerIp, "");
    node_->set_scheduler_ip(scheduler_ip);
  } else {
    node_->set_scheduler_ip(PSContext::instance()->cluster_config().scheduler_host);
  }

  // 4. recover scheduler port
  if (recovery_storage_->Exists(kRecoverySchedulerPort)) {
    uint16_t scheduler_port = std::strtol(recovery_storage_->Get(kRecoverySchedulerPort, "").c_str(), nullptr, kBase);
    node_->set_scheduler_port(scheduler_port);
  } else {
    node_->set_scheduler_port(PSContext::instance()->cluster_config().scheduler_port);
  }
  MS_LOG(INFO) << "The worker num:" << node_->worker_num() << ", the server num:" << node_->server_num()
               << ", the scheduler ip:" << node_->scheduler_ip() << ", the scheduler port:" << node_->scheduler_port();
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
