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

#ifndef MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
#define MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_

#include <string>
#include <iostream>
#include <memory>
#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
/*
 * Configuration information read through environment variables and configuration files, generally immutable
 */
struct ClusterConfig {
  explicit ClusterConfig(const uint32_t &worker_num, const uint32_t &server_num, std::string host, const uint16_t &port)
      : initial_worker_num(worker_num),
        initial_server_num(server_num),
        heartbeat_interval(3),
        scheduler_host(host),
        scheduler_port(port),
        heartbeat_timeout(30),
        cluster_available_timeout(300),
        connect_interval(3000),
        scheduler_timeout(30) {}

  // Configure through environment variables:MS_WORKER_NUM
  uint32_t initial_worker_num;
  // Configure through environment variables:MS_SERVER_NUM
  uint32_t initial_server_num;

  // The interval for sending heartbeat packets between worker node,server node and scheduler node is 3 seconds.
  uint32_t heartbeat_interval;
  std::string scheduler_host;
  uint16_t scheduler_port;
  // The timeout for worker node and server node sending heartbeat packets to scheduler node is 30 seconds.
  uint32_t heartbeat_timeout;
  // Timeout period for cluster preparation is 300 seconds.
  uint32_t cluster_available_timeout;
  // The timeout period for the client to connect to the server is 3000ms.
  uint32_t connect_interval;
  // When the scheduler exits, the worker and server can continue to work for 5 hours
  int64_t scheduler_timeout;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
