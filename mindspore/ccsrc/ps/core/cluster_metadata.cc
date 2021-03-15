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

#include "ps/core/cluster_metadata.h"
#include <string>

namespace mindspore {
namespace ps {
namespace core {
std::shared_ptr<ClusterMetadata> ClusterMetadata::instance() {
  static std::shared_ptr<ClusterMetadata> metadata_instance = nullptr;
  if (metadata_instance == nullptr) {
    metadata_instance.reset(new (std::nothrow) ClusterMetadata());
  }
  return metadata_instance;
}

void ClusterMetadata::Init(const uint32_t &worker_num, const uint32_t &server_num, std::string scheduler_host,
                           const uint16_t &scheduler_port) {
  worker_num_ = worker_num;
  server_num_ = server_num;
  scheduler_host_ = std::make_unique<std::string>(scheduler_host);
  scheduler_port_ = scheduler_port;
}

uint32_t ClusterMetadata::total_worker_num() { return worker_num_; }

uint32_t ClusterMetadata::total_server_num() { return server_num_; }

uint32_t ClusterMetadata::heartbeat_interval() { return heartbeat_interval_; }

void ClusterMetadata::set_heartbeat_interval(const uint32_t &heartbeat_interval) {
  heartbeat_interval_ = heartbeat_interval;
}

std::string ClusterMetadata::scheduler_host() {
  MS_EXCEPTION_IF_NULL(scheduler_host_);
  return *scheduler_host_;
}

uint16_t ClusterMetadata::scheduler_port() { return scheduler_port_; }

uint32_t ClusterMetadata::heartbeat_timeout() { return heartbeat_timeout_; }

void ClusterMetadata::set_heartbeat_timeout(const uint32_t &heartbeat_timeout) {
  heartbeat_interval_ = heartbeat_timeout;
}

uint32_t ClusterMetadata::cluster_available_timeout() { return cluster_available_timeout_; }

void ClusterMetadata::set_cluster_available_timeout(const uint32_t &cluster_available_timeout) {
  cluster_available_timeout_ = cluster_available_timeout;
}

uint32_t ClusterMetadata::connect_interval() { return connect_interval_; }

void ClusterMetadata::set_connect_interval(const uint32_t &connect_interval) { connect_interval_ = connect_interval; }

uint32_t ClusterMetadata::scheduler_timeout() { return scheduler_timeout_; }

void ClusterMetadata::set_scheduler_timeout(const uint32_t &scheduler_timeout) {
  scheduler_timeout_ = scheduler_timeout;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
