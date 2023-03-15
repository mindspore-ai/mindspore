/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/ps/scheduler.h"
#include "ps/core/scheduler_node.h"
#include "ps/core/ps_scheduler_node.h"

namespace mindspore {
namespace ps {
Scheduler &Scheduler::GetInstance() {
  static Scheduler instance{};
  return instance;
}

Scheduler::Scheduler() {
  if (scheduler_node_ == nullptr) {
    scheduler_node_ = std::make_unique<core::PSSchedulerNode>();
  }
}

Scheduler::~Scheduler() = default;

void Scheduler::Run() {
  MS_LOG(INFO) << "Start scheduler.";
  PSContext::instance()->cluster_config().scheduler_host = PSContext::instance()->scheduler_host();
  PSContext::instance()->cluster_config().scheduler_port = PSContext::instance()->scheduler_port();
  PSContext::instance()->cluster_config().initial_worker_num = PSContext::instance()->initial_worker_num();
  PSContext::instance()->cluster_config().initial_server_num = PSContext::instance()->initial_server_num();
  if (!scheduler_node_->Start()) {
    MS_LOG(WARNING) << "Scheduler start failed.";
  }

  if (!scheduler_node_->Finish()) {
    MS_LOG(WARNING) << "Scheduler finish failed.";
  }

  if (!scheduler_node_->Stop()) {
    MS_LOG(WARNING) << "Scheduler stop failed.";
  }
}
}  // namespace ps
}  // namespace mindspore
