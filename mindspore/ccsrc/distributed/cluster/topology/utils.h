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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_UTILS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_UTILS_H_

#include <limits.h>
#include <string>
#include <memory>
#include <chrono>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "actor/msg.h"
#include "distributed/cluster/topology/common.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
__attribute__((unused)) static bool FillMetaServerAddress(struct MetaServerAddress *address) {
  MS_EXCEPTION_IF_NULL(address);

  // Get the address of meta server from the environment.
  auto ip = common::GetEnv(kEnvMetaServerHost);
  auto ms_port = common::GetEnv(kEnvMetaServerPort);
  if (ip.empty()) {
    MS_LOG(ERROR) << "Failed to get ip of meta server from environment variables.";
    return false;
  }
  if (ms_port.empty()) {
    MS_LOG(ERROR) << "Failed to get port of meta server from environment variables.";
    return false;
  }
  auto port = std::stoi(ms_port.c_str(), nullptr, kDecimal);
  // Valid port number range.
  static const int min_port = 1;
  static const int max_port = 65535;
  if (port < min_port || port > max_port) {
    MS_LOG(ERROR) << "The port number of meta server node: " << port << " is invalid (1~65535).";
    return false;
  }

  // Fill the meta server address.
  address->ip = ip;
  address->port = port;
  return true;
}

__attribute__((unused)) static std::unique_ptr<MessageBase> CreateMessage(const std::string &dest_url,
                                                                          const std::string &name,
                                                                          const std::string &content) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  message->name = name;
  message->from = AID("", "");
  message->to = AID("", dest_url);
  message->body = content;
  return message;
}

__attribute__((unused)) static std::unique_ptr<MessageBase> CreateMessage(const std::string &dest_url,
                                                                          const MessageName &name,
                                                                          const std::string &content) {
  return CreateMessage(dest_url, std::to_string(static_cast<int>(name)), content);
}

__attribute__((unused)) static std::chrono::high_resolution_clock::time_point Now() {
  return std::chrono::high_resolution_clock::now();
}

__attribute__((unused)) static std::chrono::milliseconds ElapsedTime(
  const std::chrono::high_resolution_clock::time_point &start_time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Now() - start_time);
}

__attribute__((unused)) static bool CheckFilePath(const std::string &path) {
  char real_path[PATH_MAX] = {0};
  if (realpath(path.data(), real_path) == nullptr) {
    return false;
  }
  return true;
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_UTILS_H_
