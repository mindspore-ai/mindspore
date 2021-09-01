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

#include "ps/core/comm_util.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <regex>

namespace mindspore {
namespace ps {
namespace core {
std::random_device CommUtil::rd;
std::mt19937_64 CommUtil::gen(rd());
std::uniform_int_distribution<> CommUtil::dis = std::uniform_int_distribution<>{0, 15};
std::uniform_int_distribution<> CommUtil::dis2 = std::uniform_int_distribution<>{8, 11};

bool CommUtil::CheckIpWithRegex(const std::string &ip) {
  std::regex pattern(
    "(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])"
    "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
    "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
    "[.](25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])");
  std::smatch res;
  if (regex_match(ip, res, pattern)) {
    return true;
  }
  return false;
}

bool CommUtil::CheckIp(const std::string &ip) {
  if (!CheckIpWithRegex(ip)) {
    return false;
  }
  uint32_t uAddr = inet_addr(ip.c_str());
  if (INADDR_NONE == uAddr) {
    return false;
  }
  return true;
}

bool CommUtil::CheckPort(const uint16_t &port) {
  if (port > 65535) {
    MS_LOG(ERROR) << "The range of port should be 1 to 65535.";
    return false;
  }
  return true;
}

void CommUtil::GetAvailableInterfaceAndIP(std::string *interface, std::string *ip) {
  MS_EXCEPTION_IF_NULL(interface);
  MS_EXCEPTION_IF_NULL(ip);
  struct ifaddrs *if_address = nullptr;
  struct ifaddrs *ifa = nullptr;

  interface->clear();
  ip->clear();
  if (getifaddrs(&if_address) == -1) {
    MS_LOG(WARNING) << "Get ifaddrs failed.";
  }
  for (ifa = if_address; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    if (ifa->ifa_addr->sa_family == AF_INET && (ifa->ifa_flags & IFF_LOOPBACK) == 0) {
      char address_buffer[INET_ADDRSTRLEN] = {0};
      void *sin_addr_ptr = &(reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr))->sin_addr;
      MS_EXCEPTION_IF_NULL(sin_addr_ptr);
      const char *net_ptr = inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);
      MS_EXCEPTION_IF_NULL(net_ptr);

      *ip = address_buffer;
      *interface = ifa->ifa_name;
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(if_address);
  freeifaddrs(if_address);
}

std::string CommUtil::GenerateUUID() {
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < kGroup1RandomLength; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < kGroup2RandomLength; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < kGroup3RandomLength - 1; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < kGroup4RandomLength - 1; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < kGroup5RandomLength; i++) {
    ss << dis(gen);
  }
  return ss.str();
}

std::string CommUtil::NodeRoleToString(const NodeRole &role) {
  switch (role) {
    case NodeRole::SCHEDULER:
      return "SCHEDULER";
    case NodeRole::SERVER:
      return "SERVER";
    case NodeRole::WORKER:
      return "WORKER";
    default:
      MS_LOG(EXCEPTION) << "The node role:" << role << " is illegal!";
  }
}
bool CommUtil::ValidateRankId(const enum NodeRole &node_role, const uint32_t &rank_id, const int32_t &total_worker_num,
                              const int32_t &total_server_num) {
  if (node_role == NodeRole::SERVER && (rank_id > IntToUint(total_server_num) - 1)) {
    return false;
  } else if (node_role == NodeRole::WORKER && (rank_id > IntToUint(total_worker_num) - 1)) {
    return false;
  }
  return true;
}

bool CommUtil::Retry(const std::function<bool()> &func, size_t max_attempts, size_t interval_milliseconds) {
  for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
    if (func()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_milliseconds));
  }
  return false;
}

void CommUtil::LogCallback(int severity, const char *msg) {
  MS_EXCEPTION_IF_NULL(msg);
  switch (severity) {
    case EVENT_LOG_MSG:
      MS_LOG(INFO) << kLibeventLogPrefix << msg;
      break;
    case EVENT_LOG_WARN:
      MS_LOG(WARNING) << kLibeventLogPrefix << msg;
      break;
    case EVENT_LOG_ERR:
      MS_LOG(ERROR) << kLibeventLogPrefix << msg;
      break;
    default:
      break;
  }
}

bool CommUtil::IsFileExists(const std::string &file) {
  std::ifstream f(file.c_str());
  if (!f.good()) {
    return false;
  } else {
    f.close();
    return true;
  }
}

std::string CommUtil::ClusterStateToString(const ClusterState &state) {
  MS_LOG(INFO) << "The cluster state:" << state;
  if (state < SizeToInt(kClusterState.size())) {
    return kClusterState.at(state);
  } else {
    return "";
  }
}

std::string CommUtil::ParseConfig(const Configuration &config, const std::string &data) {
  if (!config.IsInitialized()) {
    MS_LOG(INFO) << "The config is not initialized.";
    return "";
  }

  if (!const_cast<Configuration &>(config).Exists(data)) {
    MS_LOG(INFO) << "The data:" << data << " is not exist.";
    return "";
  }

  std::string path = config.GetString(data, "");
  return path;
}

bool CommUtil::VerifyCertTime(const X509 *cert, int64_t time) {
  MS_EXCEPTION_IF_NULL(cert);
  ASN1_TIME *start = X509_getm_notBefore(cert);
  ASN1_TIME *end = X509_getm_notAfter(cert);
  MS_EXCEPTION_IF_NULL(start);
  MS_EXCEPTION_IF_NULL(end);
  int day = 0;
  int sec = 0;
  if (!ASN1_TIME_diff(&day, &sec, start, NULL)) {
    MS_LOG(WARNING) << "ASN1 time diff failed.";
    return false;
  }

  if (day < 0 || sec < 0) {
    MS_LOG(WARNING) << "Cert start time is later than now time.";
    return false;
  }
  day = 0;
  sec = 0;

  if (!ASN1_TIME_diff(&day, &sec, NULL, end)) {
    MS_LOG(WARNING) << "ASN1 time diff failed.";
    return false;
  }

  int64_t interval = kCertExpireWarningTimeInDay;
  if (time > 0) {
    interval = time;
  }

  if (day < LongToInt(interval) && day >= 0) {
    MS_LOG(WARNING) << "The certificate will expire in " << day << " days and " << sec << " seconds.";
  } else if (day < 0 || sec < 0) {
    MS_LOG(WARNING) << "The certificate has expired.";
    return false;
  }
  return true;
}

bool CommUtil::VerifyCRL(const X509 *cert, const std::string &crl_path) {
  MS_ERROR_IF_NULL_W_RET_VAL(cert, false);
  BIO *bio = BIO_new_file(crl_path.c_str(), "r");
  MS_ERROR_IF_NULL_W_RET_VAL(bio, false);
  X509_CRL *root_crl = PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr);
  MS_ERROR_IF_NULL_W_RET_VAL(root_crl, false);
  EVP_PKEY *evp_pkey = X509_get_pubkey(const_cast<X509 *>(cert));
  MS_ERROR_IF_NULL_W_RET_VAL(evp_pkey, false);

  int ret = X509_CRL_verify(root_crl, evp_pkey);
  BIO_free_all(bio);
  if (ret == 1) {
    MS_LOG(WARNING) << "Equip cert in root crl, verify failed";
    return false;
  }
  MS_LOG(INFO) << "VerifyCRL success.";
  return true;
}

bool CommUtil::VerifyCommonName(const X509 *cert, const std::string &ca_path) {
  MS_ERROR_IF_NULL_W_RET_VAL(cert, false);
  X509 *cert_temp = const_cast<X509 *>(cert);
  char subject_cn[256] = "";
  char issuer_cn[256] = "";
  X509_NAME *subject_name = X509_get_subject_name(cert_temp);
  X509_NAME *issuer_name = X509_get_issuer_name(cert_temp);
  MS_ERROR_IF_NULL_W_RET_VAL(subject_name, false);
  MS_ERROR_IF_NULL_W_RET_VAL(issuer_name, false);
  if (!X509_NAME_get_text_by_NID(subject_name, NID_commonName, subject_cn, sizeof(subject_cn))) {
    MS_LOG(WARNING) << "Get text by nid failed.";
    return false;
  }
  if (!X509_NAME_get_text_by_NID(issuer_name, NID_commonName, issuer_cn, sizeof(issuer_cn))) {
    MS_LOG(WARNING) << "Get text by nid failed.";
    return false;
  }
  MS_LOG(INFO) << "the subject:" << subject_cn << ", the issuer:" << issuer_cn;

  BIO *ca_bio = BIO_new_file(ca_path.c_str(), "r");
  MS_EXCEPTION_IF_NULL(ca_bio);
  X509 *ca_cert = PEM_read_bio_X509(ca_bio, nullptr, nullptr, nullptr);
  MS_EXCEPTION_IF_NULL(ca_cert);
  char ca_subject_cn[256] = "";
  char ca_issuer_cn[256] = "";
  X509_NAME *ca_subject_name = X509_get_subject_name(ca_cert);
  X509_NAME *ca_issuer_name = X509_get_issuer_name(ca_cert);
  MS_ERROR_IF_NULL_W_RET_VAL(ca_subject_name, false);
  MS_ERROR_IF_NULL_W_RET_VAL(ca_issuer_name, false);
  if (!X509_NAME_get_text_by_NID(ca_subject_name, NID_commonName, ca_subject_cn, sizeof(subject_cn))) {
    MS_LOG(WARNING) << "Get text by nid failed.";
    return false;
  }
  if (!X509_NAME_get_text_by_NID(ca_issuer_name, NID_commonName, ca_issuer_cn, sizeof(issuer_cn))) {
    MS_LOG(WARNING) << "Get text by nid failed.";
    return false;
  }
  MS_LOG(INFO) << "the subject:" << ca_subject_cn << ", the issuer:" << ca_issuer_cn;
  BIO_free_all(ca_bio);
  if (strcmp(issuer_cn, ca_subject_cn) != 0) {
    return false;
  }
  return true;
}

std::vector<std::string> CommUtil::Split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

bool CommUtil::VerifyCipherList(const std::vector<std::string> &list) {
  for (auto &item : list) {
    if (!kCiphers.count(item)) {
      MS_LOG(WARNING) << "The ciphter:" << item << " is not supported.";
      return false;
    }
  }
  return true;
}

void CommUtil::InitOpenSSLEnv() {
  if (!SSL_library_init()) {
    MS_LOG(EXCEPTION) << "SSL_library_init failed.";
  }
  if (!ERR_load_crypto_strings()) {
    MS_LOG(EXCEPTION) << "ERR_load_crypto_strings failed.";
  }
  if (!SSL_load_error_strings()) {
    MS_LOG(EXCEPTION) << "SSL_load_error_strings failed.";
  }
  if (!OpenSSL_add_all_algorithms()) {
    MS_LOG(EXCEPTION) << "OpenSSL_add_all_algorithms failed.";
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
