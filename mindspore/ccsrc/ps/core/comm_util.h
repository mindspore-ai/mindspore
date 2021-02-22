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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_
#define MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_

#include <unistd.h>
#ifdef _MSC_VER
#include <tchar.h>
#include <winsock2.h>
#include <windows.h>
#include <iphlpapi.h>
#else
#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#endif

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <thread>

#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/core/cluster_metadata.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr int kGroup1RandomLength = 8;
constexpr int kGroup2RandomLength = 4;
constexpr int kGroup3RandomLength = 4;
constexpr int kGroup4RandomLength = 4;
constexpr int kGroup5RandomLength = 12;

// The size of the buffer for sending and receiving data is 4096 bytes.
constexpr int kMessageChunkLength = 4096;
// The timeout period for the http client to connect to the http server is 120 seconds.
constexpr int kConnectionTimeout = 120;
constexpr char kLibeventLogPrefix[] = "[libevent log]:";

class CommUtil {
 public:
  static bool CheckIpWithRegex(const std::string &ip);
  static bool CheckIp(const std::string &ip);
  static void GetAvailableInterfaceAndIP(std::string *interface, std::string *ip);
  static std::string GenerateUUID();
  static std::string NodeRoleToString(const NodeRole &role);
  static bool ValidateRankId(const enum NodeRole &node_role, const uint32_t &rank_id);
  static bool Retry(const std::function<bool()> &func, size_t max_attempts, size_t interval_milliseconds);
  static void LogCallback(int severity, const char *msg);

 private:
  static std::random_device rd;
  static std::mt19937_64 gen;
  static std::uniform_int_distribution<> dis;
  static std::uniform_int_distribution<> dis2;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_
