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

#ifndef MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
#define MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_

#include <string>
#include <iostream>
#include <memory>
#include <utility>

#include "utils/log_adapter.h"
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
class ClusterConfig {
 public:
  static void Init(const uint32_t &worker_num, const uint32_t &server_num, std::string scheduler_host,
                   const uint16_t &scheduler_port);
  static uint32_t worker_num();
  static uint32_t server_num();
  static uint32_t heartbeat_interval();
  static void set_heartbeat_interval(const uint32_t &heartbeat_interval);
  static std::string scheduler_host();
  static uint16_t scheduler_port();
  static uint32_t heartbeat_timeout();
  static void set_heartbeat_timeout(const uint32_t &heartbeat_timeout);
  static uint32_t cluster_available_timeout();
  static void set_cluster_available_timeout(const uint32_t &cluster_available_timeout);
  static uint32_t connect_interval();
  static void set_connect_interval(const uint32_t &connect_interval);
  static uint32_t scheduler_timeout();
  static void set_scheduler_timeout(const uint32_t &scheduler_timeout);

 private:
  static uint32_t worker_num_;
  static uint32_t server_num_;
  static uint32_t heartbeat_interval_;
  static std::unique_ptr<std::string> scheduler_host_;
  static uint16_t scheduler_port_;
  static uint32_t heartbeat_timeout_;
  static uint32_t cluster_available_timeout_;
  static uint32_t connect_interval_;
  static uint32_t scheduler_timeout_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
