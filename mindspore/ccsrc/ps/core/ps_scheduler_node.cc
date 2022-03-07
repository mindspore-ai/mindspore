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

#include <memory>
#include "ps/core/ps_scheduler_node.h"

namespace mindspore {
namespace ps {
namespace core {
void PSSchedulerNode::RunRecovery() {
  const auto &clusterConfig = PSContext::instance()->cluster_config();
  // create tcp client to myself in case of event dispatch failed when Send reconnect msg to server failed
  client_to_scheduler_ =
    std::make_shared<TcpClient>(clusterConfig.scheduler_host, clusterConfig.scheduler_port, NodeRole::SCHEDULER);
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  client_to_scheduler_->Init();
  client_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
  MS_EXCEPTION_IF_NULL(client_thread_);

  const auto &initial_node_infos = clusterConfig.initial_registered_nodes_infos;
  if (initial_node_infos.empty()) {
    MS_LOG(WARNING) << "There is no registered nodes in scheduler!";
    return;
  }
  MS_LOG(INFO) << "The scheduler start run recovery!";
  uint32_t worker_num = clusterConfig.initial_worker_num;
  uint32_t server_num = clusterConfig.initial_server_num;

  node_manager_.set_worker_num(worker_num);
  node_manager_.set_server_num(server_num);
  node_manager_.set_next_worker_rank_id(clusterConfig.initial_next_worker_rank_id);
  node_manager_.set_next_server_rank_id(clusterConfig.initial_next_server_rank_id);
  node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);

  MS_LOG(INFO) << "Scheduler recovery finish.";
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
