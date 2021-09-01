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

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <assert.h>
#include <openssl/pkcs12.h>
#include <openssl/bio.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <thread>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "utils/log_adapter.h"
#include "ps/ps_context.h"
#include "utils/convert_utils_base.h"
#include "ps/core/configuration.h"

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

// Find the corresponding string style of cluster state through the subscript of the enum:ClusterState
const std::vector<std::string> kClusterState = {
  "ClUSTER_STARTING",   // Initialization state when the cluster is just started.
  "CLUSTER_READY",      // The state after all nodes are successfully registered.
  "CLUSTER_EXIT",       // The state after the cluster exits successfully.
  "NODE_TIMEOUT",       // When a node has a heartbeat timeout
  "CLUSTER_SCALE_OUT",  // When the cluster is scale out.
  "CLUSTER_SCALE_IN"    // When the cluster is scale in.
};

class CommUtil {
 public:
  static bool CheckIpWithRegex(const std::string &ip);
  static bool CheckIp(const std::string &ip);
  static bool CheckPort(const uint16_t &port);
  static void GetAvailableInterfaceAndIP(std::string *interface, std::string *ip);
  static std::string GenerateUUID();
  static std::string NodeRoleToString(const NodeRole &role);
  static bool ValidateRankId(const enum NodeRole &node_role, const uint32_t &rank_id, const int32_t &total_worker_num,
                             const int32_t &total_server_num);
  static bool Retry(const std::function<bool()> &func, size_t max_attempts, size_t interval_milliseconds);
  static void LogCallback(int severity, const char *msg);

  // Check if the file exists.
  static bool IsFileExists(const std::string &file);
  // Convert cluster state to string when response the http request.
  static std::string ClusterStateToString(const ClusterState &state);

  // Parse the configuration file according to the key.
  static std::string ParseConfig(const Configuration &config, const std::string &data);

  // verify valid of certificate time
  static bool VerifyCertTime(const X509 *cert, int64_t time = 0);
  // verify valid of equip certificate with CRL
  static bool VerifyCRL(const X509 *cert, const std::string &crl_path);
  // Check the common name of the certificate
  static bool VerifyCommonName(const X509 *cert, const std::string &ca_path);
  // The string is divided according to delim
  static std::vector<std::string> Split(const std::string &s, char delim);
  // Check the cipher list of the certificate
  static bool VerifyCipherList(const std::vector<std::string> &list);
  static void InitOpenSSLEnv();

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
