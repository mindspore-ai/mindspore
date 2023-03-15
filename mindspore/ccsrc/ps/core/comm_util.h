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
#include <iphlpapi.h>
#include <tchar.h>
#include <windows.h>
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#endif

#include <assert.h>
#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pkcs12.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "utils/log_adapter.h"
#include "include/backend/distributed/ps/ps_context.h"
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
constexpr int kMillSecondLength = 3;

// The size of the buffer for sending and receiving data is 4096 bytes.
constexpr int kMessageChunkLength = 4096;
// The timeout period for the http client to connect to the http server is 120 seconds.
constexpr int kConnectionTimeout = 120;
constexpr char kLibeventLogPrefix[] = "[libevent log]:";
constexpr char kFailureEvent[] = "failureEvent";

// Find the corresponding string style of cluster state through the subscript of the enum:ClusterState
const std::vector<std::string> kClusterState = {
  "CLUSTER_STARTING",            // Initialization state when the cluster is just started.
  "CLUSTER_READY",               // The state after all nodes are successfully registered.
  "CLUSTER_EXIT",                // The state after the cluster exits successfully.
  "NODE_TIMEOUT",                // When a node has a heartbeat timeout
  "CLUSTER_SCALE_OUT",           // When the cluster is scale out.
  "CLUSTER_SCALE_IN",            // When the cluster is scale in.
  "CLUSTER_NEW_INSTANCE",        // When the cluster is doing NEW_INSTANCE.
  "CLUSTER_ENABLE_FLS",          // When the cluster is doing ENABLE_FLS.
  "CLUSTER_DISABLE_FLS",         // When the cluster is doing DISABLE_FLS.
  "CLUSTER_SCHEDULER_RECOVERY",  // When the cluster is doing SCHEDULER_RECOVERY.
  "CLUSTER_SCALE_OUT_ROLLBACK",  // When the cluster is scale out rollback.
};

const std::map<std::string, ClusterState> kClusterStateMap = {
  {"CLUSTER_STARTING", ClusterState::CLUSTER_STARTING},
  {"CLUSTER_READY", ClusterState::CLUSTER_READY},
  {"CLUSTER_EXIT", ClusterState::CLUSTER_EXIT},
  {"NODE_TIMEOUT", ClusterState::NODE_TIMEOUT},
  {"CLUSTER_SCALE_OUT", ClusterState::CLUSTER_SCALE_OUT},
  {"CLUSTER_SCALE_IN", ClusterState::CLUSTER_SCALE_IN},
  {"CLUSTER_NEW_INSTANCE", ClusterState::CLUSTER_NEW_INSTANCE},
  {"CLUSTER_ENABLE_FLS", ClusterState::CLUSTER_ENABLE_FLS},
  {"CLUSTER_DISABLE_FLS", ClusterState::CLUSTER_DISABLE_FLS},
  {"CLUSTER_SCHEDULER_RECOVERY", ClusterState::CLUSTER_SCHEDULER_RECOVERY},
  {"CLUSTER_SCALE_OUT_ROLLBACK", ClusterState::CLUSTER_SCALE_OUT_ROLLBACK}};

struct Time {
  uint64_t time_stamp;
  std::string time_str_mill;
};

struct FileConfig {
  uint32_t storage_type;
  std::string storage_file_path;
};

class CommUtil {
 public:
  static bool CheckIpWithRegex(const std::string &ip);
  static bool CheckIp(const std::string &ip);
  static bool CheckPort(const uint16_t &port);
  static void GetAvailableInterfaceAndIP(std::string *interface, std::string *ip);
  static std::string GetLoopBackInterfaceName();
  static std::string GenerateUUID();
  static std::string NodeRoleToString(const NodeRole &role);
  static NodeRole StringToNodeRole(const std::string &roleStr);
  static std::string BoolToString(bool alive);
  static bool StringToBool(const std::string &alive);
  static bool ValidateRankId(const enum NodeRole &node_role, const uint32_t &rank_id, const int32_t &total_worker_num,
                             const int32_t &total_server_num);
  static bool Retry(const std::function<bool()> &func, size_t max_attempts, size_t interval_milliseconds);
  static void LogCallback(int severity, const char *msg);

  // Check if the file exists.
  static bool IsFileExists(const std::string &file);
  // Check whether the file is empty or not.
  static bool IsFileEmpty(const std::string &file);
  // Convert cluster state to string when response the http request.
  static std::string ClusterStateToString(const ClusterState &state);

  // Parse the configuration file according to the key.
  static std::string ParseConfig(const Configuration &config, const std::string &key);

  // Init openssl lib
  static void InitOpensslLib();

  // verify valid of certificate time
  static bool VerifyCertTime(const X509 *cert, int64_t time = 0);
  static bool verifyCertTimeStamp(const X509 *cert);
  // verify valid of equip certificate with CRL
  static bool VerifyCRL(const X509 *cert, const std::string &crl_path, X509_CRL **crl);
  static bool VerifyCommonName(const X509 *caCert, const X509 *subCert);
  static std::vector<std::string> Split(const std::string &s, char delim);
  static bool VerifyCipherList(const std::vector<std::string> &list);
  static bool verifyCertKeyID(const X509 *caCert, const X509 *subCert);
  static bool verifySingature(const X509 *caCert, const X509 *subCert);
  static bool verifyExtendedAttributes(const X509 *caCert);
  static void verifyCertPipeline(const X509 *caCert, const X509 *subCert);
  static bool checkCRLTime(const std::string &crlPath);
  static bool CreateDirectory(const std::string &directoryPath);
  static bool CheckHttpUrl(const std::string &http_url);
  static bool IsFileReadable(const std::string &file);
  template <typename T>
  static T JsonGetKeyWithException(const nlohmann::json &json, const std::string &key) {
    if (!json.contains(key)) {
      MS_LOG(EXCEPTION) << "The key " << key << "does not exist in json " << json.dump();
    }
    return json[key].get<T>();
  }
  static Time GetNowTime();
  static bool ParseAndCheckConfigJson(Configuration *file_configuration, const std::string &key,
                                      FileConfig *file_config);

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
