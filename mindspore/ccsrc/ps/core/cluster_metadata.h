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

#ifndef MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_
#define MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_

#include <string>
#include <iostream>
#include <memory>
#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
class ClusterMetadata {
 public:
  ~ClusterMetadata() = default;
  ClusterMetadata(ClusterMetadata const &) = delete;
  ClusterMetadata &operator=(const ClusterMetadata &) = delete;
  static std::shared_ptr<ClusterMetadata> instance();

  void Init(const uint32_t &worker_num, const uint32_t &server_num, std::string scheduler_host,
            const uint16_t &scheduler_port);
  uint32_t total_worker_num();
  uint32_t total_server_num();
  uint32_t heartbeat_interval();
  void set_heartbeat_interval(const uint32_t &heartbeat_interval);
  std::string scheduler_host();
  uint16_t scheduler_port();
  uint32_t heartbeat_timeout();
  void set_heartbeat_timeout(const uint32_t &heartbeat_timeout);
  uint32_t cluster_available_timeout();
  void set_cluster_available_timeout(const uint32_t &cluster_available_timeout);
  uint32_t connect_interval();
  void set_connect_interval(const uint32_t &connect_interval);
  uint32_t scheduler_timeout();
  void set_scheduler_timeout(const uint32_t &scheduler_timeout);

 private:
  ClusterMetadata()
      : worker_num_(0),
        server_num_(0),
        heartbeat_interval_(3),
        scheduler_host_(nullptr),
        scheduler_port_(0),
        heartbeat_timeout_(30),
        cluster_available_timeout_(300),
        connect_interval_(100),
        scheduler_timeout_(30) {}
  uint32_t worker_num_;
  uint32_t server_num_;
  // The interval for sending heartbeat packets between worker node,server node and scheduler node is 3 seconds.
  uint32_t heartbeat_interval_;
  std::unique_ptr<std::string> scheduler_host_;
  uint16_t scheduler_port_;
  // The timeout for worker node and server node sending heartbeat packets to scheduler node is 30 seconds.
  uint32_t heartbeat_timeout_;
  // Timeout period for cluster preparation is 300 seconds.
  uint32_t cluster_available_timeout_;
  // The timeout period for the client to connect to the server is 100ms.
  uint32_t connect_interval_;
  // When the scheduler exits, the worker and server can continue to work for 5 hours
  uint32_t scheduler_timeout_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_
