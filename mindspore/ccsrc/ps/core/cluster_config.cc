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

#include "ps/core/cluster_config.h"

#include <string>

namespace mindspore {
namespace ps {
namespace core {
uint32_t ClusterConfig::worker_num_ = 0;
uint32_t ClusterConfig::server_num_ = 0;
std::unique_ptr<std::string> ClusterConfig::scheduler_host_ = nullptr;
uint16_t ClusterConfig::scheduler_port_ = 0;
// The interval for sending heartbeat packets between worker node,server node and scheduler node is 3 seconds.
uint32_t ClusterConfig::heartbeat_interval_ = 3;
// The timeout for worker node and server node sending heartbeat packets to scheduler node is 30 seconds.
uint32_t ClusterConfig::heartbeat_timeout_ = 30;
// Timeout period for cluster preparation is 300 seconds.
uint32_t ClusterConfig::cluster_available_timeout_ = 300;
// The timeout period for the client to connect to the server is 100ms.
uint32_t ClusterConfig::connect_interval_ = 100;
// When the scheduler exits, the worker and server can continue to work for 5 hours
uint32_t ClusterConfig::scheduler_timeout_ = 3600 * 5;

void ClusterConfig::Init(const uint32_t &worker_num, const uint32_t &server_num, std::string scheduler_host,
                         const uint16_t &scheduler_port) {
  worker_num_ = worker_num;
  server_num_ = server_num;
  if (!CommUtil::CheckIp(scheduler_host)) {
    MS_LOG(EXCEPTION) << "The scheduler_host:" << scheduler_host << " is illegal!";
  }
  scheduler_host_ = std::make_unique<std::string>(scheduler_host);
  scheduler_port_ = scheduler_port;
}

uint32_t ClusterConfig::worker_num() { return worker_num_; }

uint32_t ClusterConfig::server_num() { return server_num_; }

uint32_t ClusterConfig::heartbeat_interval() { return heartbeat_interval_; }

void ClusterConfig::set_heartbeat_interval(const uint32_t &heartbeat_interval) {
  heartbeat_interval_ = heartbeat_interval;
}

std::string ClusterConfig::scheduler_host() { return *scheduler_host_; }

uint16_t ClusterConfig::scheduler_port() { return scheduler_port_; }

uint32_t ClusterConfig::heartbeat_timeout() { return heartbeat_timeout_; }

void ClusterConfig::set_heartbeat_timeout(const uint32_t &heartbeat_timeout) {
  heartbeat_interval_ = heartbeat_timeout;
}

uint32_t ClusterConfig::cluster_available_timeout() { return cluster_available_timeout_; }

void ClusterConfig::set_cluster_available_timeout(const uint32_t &cluster_available_timeout) {
  cluster_available_timeout_ = cluster_available_timeout;
}

uint32_t ClusterConfig::connect_interval() { return connect_interval_; }

void ClusterConfig::set_connect_interval(const uint32_t &connect_interval) { connect_interval_ = connect_interval; }

uint32_t ClusterConfig::scheduler_timeout() { return scheduler_timeout_; }

void ClusterConfig::set_scheduler_timeout(const uint32_t &scheduler_timeout) { scheduler_timeout_ = scheduler_timeout; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
