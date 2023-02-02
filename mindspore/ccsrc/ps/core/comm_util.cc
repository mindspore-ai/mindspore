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
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
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
    "(([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\\.)"
    "{3}([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])");
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

bool CommUtil::CheckHttpUrl(const std::string &http_url) {
  std::regex pattern(
    "https?:\\/\\/(www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_\\+.~#?&//=]*)");
  std::smatch res;
  if (regex_match(http_url, res, pattern)) {
    return true;
  }
  return false;
}

bool CommUtil::CheckPort(const uint16_t &port) {
  if (port > kMaxPort) {
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

std::string CommUtil::GetLoopBackInterfaceName() {
  struct ifaddrs *if_address = nullptr;
  struct ifaddrs *ifa = nullptr;

  if (getifaddrs(&if_address) == -1) {
    MS_LOG(WARNING) << "Get ifaddrs failed.";
  }
  for (ifa = if_address; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    if (ifa->ifa_flags & IFF_LOOPBACK) {
      MS_LOG(INFO) << "Loop back interface name is " << ifa->ifa_name;
      return ifa->ifa_name;
    }
  }
  MS_EXCEPTION_IF_NULL(if_address);
  freeifaddrs(if_address);
  return "";
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

NodeRole CommUtil::StringToNodeRole(const std::string &roleStr) {
  if (roleStr == "SCHEDULER") {
    return NodeRole::SCHEDULER;
  } else if (roleStr == "SERVER") {
    return NodeRole::SERVER;
  } else if (roleStr == "WORKER") {
    return NodeRole::WORKER;
  } else {
    MS_LOG(EXCEPTION) << "The node role string:" << roleStr << " is illegal!";
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

bool CommUtil::IsFileExists(const std::string &file) { return access(file.c_str(), F_OK) != -1; }

bool CommUtil::IsFileReadable(const std::string &file) { return access(file.c_str(), R_OK) != -1; }

bool CommUtil::IsFileEmpty(const std::string &file) {
  if (!IsFileExists(file)) {
    MS_LOG(EXCEPTION) << "The file does not exist, file path: " << file;
  }

  std::ifstream fs(file.c_str());
  std::string str;
  fs >> str;
  fs.close();

  return str.empty();
}

bool CommUtil::CreateDirectory(const std::string &directoryPath) {
  uint32_t dirPathLen = SizeToUint(directoryPath.length());
  constexpr uint32_t MAX_PATH_LEN = 512;
  if (dirPathLen > MAX_PATH_LEN) {
    return false;
  }
  char tmpDirPath[MAX_PATH_LEN] = {0};
  for (uint32_t i = 0; i < dirPathLen; ++i) {
    tmpDirPath[i] = directoryPath[i];
    if (tmpDirPath[i] == '/') {
      if (access(tmpDirPath, 0) != 0) {
        int32_t ret = mkdir(tmpDirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (ret != 0) {
          return false;
        }
      }
    }
  }
  return true;
}

std::string CommUtil::ClusterStateToString(const ClusterState &state) {
  if (state < SizeToInt(kClusterState.size())) {
    return kClusterState.at(state);
  } else {
    return std::to_string(state);
  }
}

std::string CommUtil::ParseConfig(const Configuration &config, const std::string &key) {
  if (!config.IsInitialized()) {
    MS_LOG(INFO) << "The config is not initialized.";
    return "";
  }

  if (!const_cast<Configuration &>(config).Exists(key)) {
    MS_LOG(INFO) << "The key:" << key << " is not exist.";
    return "";
  }

  std::string path = config.GetString(key, "");
  return path;
}

bool CommUtil::verifyCertTimeStamp(const X509 *cert) {
  ASN1_TIME *start = X509_getm_notBefore(cert);
  ASN1_TIME *end = X509_getm_notAfter(cert);

  int day = 0;
  int sec = 0;
  int ret = ASN1_TIME_diff(&day, &sec, start, NULL);
  if (ret != 1) {
    return false;
  }

  if (day < 0 || sec < 0) {
    MS_LOG(ERROR) << "cert start time is later than now time.";
    return false;
  }
  day = 0;
  sec = 0;
  ret = ASN1_TIME_diff(&day, &sec, NULL, end);
  if (ret != 1) {
    return false;
  }

  if (day < 0 || sec < 0) {
    MS_LOG(ERROR) << "cert end time is sooner than now time.";
    return false;
  }
  return true;
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

bool CommUtil::VerifyCRL(const X509 *cert, const std::string &crl_path, X509_CRL **crl) {
  MS_ERROR_IF_NULL_W_RET_VAL(cert, false);
  MS_ERROR_IF_NULL_W_RET_VAL(crl, false);

  EVP_PKEY *evp_pkey = X509_get_pubkey(const_cast<X509 *>(cert));
  MS_ERROR_IF_NULL_W_RET_VAL(evp_pkey, false);
  BIO *bio = BIO_new_file(crl_path.c_str(), "r");
  MS_ERROR_IF_NULL_W_RET_VAL(bio, false);
  *crl = PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr);
  MS_ERROR_IF_NULL_W_RET_VAL(*crl, false);

  int ret = X509_CRL_verify(*crl, evp_pkey);
  if (ret == 1) {
    MS_LOG(INFO) << "VerifyCRL success.";
  } else if (ret == 0) {
    MS_LOG(ERROR) << "Verify CRL failed.";
  } else {
    MS_LOG(ERROR) << "CRL cannot be verified.";
  }
  BIO_free_all(bio);
  EVP_PKEY_free(evp_pkey);
  return ret == 1;
}

bool CommUtil::VerifyCommonName(const X509 *caCert, const X509 *subCert) {
  MS_EXCEPTION_IF_NULL(caCert);
  MS_EXCEPTION_IF_NULL(subCert);
  char caSubjectCN[256] = "";
  char subIssuerCN[256] = "";

  X509_NAME *caSubjectX509CN = X509_get_subject_name(caCert);
  X509_NAME *subIssuerX509CN = X509_get_issuer_name(subCert);

  int ret = X509_NAME_get_text_by_NID(caSubjectX509CN, NID_commonName, caSubjectCN, sizeof(caSubjectCN));
  if (ret < 0) {
    return false;
  }
  ret = X509_NAME_get_text_by_NID(subIssuerX509CN, NID_commonName, subIssuerCN, sizeof(subIssuerCN));
  if (ret < 0) {
    return false;
  }

  std::string caSubjectCNStr = caSubjectCN;
  std::string subIssuerCNStr = subIssuerCN;

  if (caSubjectCNStr != subIssuerCNStr) {
    MS_LOG(EXCEPTION) << "root CA cert subject cn is not equal with equip CA cert issuer cn.";
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

bool CommUtil::verifyCertKeyID(const X509 *caCert, const X509 *subCert) {
  MS_EXCEPTION_IF_NULL(caCert);
  MS_EXCEPTION_IF_NULL(subCert);
  int crit = 0;
  ASN1_OCTET_STRING *skid =
    reinterpret_cast<ASN1_OCTET_STRING *>(X509_get_ext_d2i(caCert, NID_subject_key_identifier, &crit, NULL));
  MS_EXCEPTION_IF_NULL(skid);
  const size_t keyidLen = 512;
  char subject_keyid[keyidLen] = {0};
  for (int i = 0; i < skid->length; i++) {
    char keyid[8] = {0};
    size_t base = keyidLen;
    if (sprintf_s(keyid, sizeof(keyid), "%x ", (uint32_t)skid->data[i]) == -1) {
      return false;
    }
    errno_t ret = strcat_s(subject_keyid, base, keyid);
    if (ret != EOK) {
      return false;
    }
  }

  AUTHORITY_KEYID *akeyid =
    reinterpret_cast<AUTHORITY_KEYID *>(X509_get_ext_d2i(subCert, NID_authority_key_identifier, &crit, NULL));
  MS_EXCEPTION_IF_NULL(akeyid);
  MS_EXCEPTION_IF_NULL(akeyid->keyid);
  char issuer_keyid[keyidLen] = {0};
  for (int i = 0; i < akeyid->keyid->length; i++) {
    char keyid[8] = {0};
    size_t base = keyidLen;
    if (sprintf_s(keyid, sizeof(keyid), "%x ", (uint32_t)(akeyid->keyid->data[i])) == -1) {
      return false;
    }
    int ret = strcat_s(issuer_keyid, base, keyid);
    if (ret != EOK) {
      return false;
    }
  }

  std::string subject_keyid_str = subject_keyid;
  std::string issuer_keyid_str = issuer_keyid;
  if (subject_keyid_str != issuer_keyid_str) {
    return false;
  }
  return true;
}

bool CommUtil::verifySingature(const X509 *caCert, const X509 *subCert) {
  MS_EXCEPTION_IF_NULL(caCert);
  MS_EXCEPTION_IF_NULL(subCert);
  EVP_PKEY *caCertPubKey = X509_get_pubkey(const_cast<X509 *>(caCert));

  int ret = 0;
  ret = X509_verify(const_cast<X509 *>(subCert), caCertPubKey);
  if (ret != 1) {
    EVP_PKEY_free(caCertPubKey);
    MS_LOG(ERROR) << "sub cert verify is failed, error code " << ret;
    return false;
  }
  MS_LOG(INFO) << "verifyCAChain success.";
  EVP_PKEY_free(caCertPubKey);
  return true;
}

bool CommUtil::verifyExtendedAttributes(const X509 *caCert) {
  MS_EXCEPTION_IF_NULL(caCert);
  int cirt = 0;
  BASIC_CONSTRAINTS *bcons =
    reinterpret_cast<BASIC_CONSTRAINTS *>(X509_get_ext_d2i(caCert, NID_basic_constraints, &cirt, nullptr));
  if (bcons == nullptr) {
    return false;
  }
  if (!bcons->ca) {
    MS_LOG(ERROR) << "Subject Type is End Entity.";
    return false;
  }
  MS_LOG(INFO) << "Subject Type is CA.";
  return true;
}

void CommUtil::InitOpensslLib() {
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

void CommUtil::verifyCertPipeline(const X509 *caCert, const X509 *subCert) {
  if (!CommUtil::VerifyCommonName(caCert, subCert)) {
    MS_LOG(EXCEPTION) << "Verify common name failed.";
  }

  if (!CommUtil::verifySingature(caCert, subCert)) {
    MS_LOG(EXCEPTION) << "Verify Signature failed.";
  }

  if (!CommUtil::verifyExtendedAttributes(caCert)) {
    MS_LOG(EXCEPTION) << "Verify Extended Attributes failed.";
  }

  if (!CommUtil::verifyCertKeyID(caCert, subCert)) {
    MS_LOG(EXCEPTION) << "Verify Cert KeyID failed.";
  }

  if (!CommUtil::verifyCertTimeStamp(caCert) || !CommUtil::verifyCertTimeStamp(subCert)) {
    MS_LOG(EXCEPTION) << "Verify Cert Time failed.";
  }
}

bool CommUtil::checkCRLTime(const std::string &crlPath) {
  if (!IsFileExists(crlPath)) {
    return true;
  }
  BIO *bio = BIO_new_file(crlPath.c_str(), "r");
  if (bio == nullptr) {
    return true;
  }
  bool result = true;
  X509_CRL *crl = nullptr;
  do {
    crl = PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr);
    if (crl == nullptr) {
      MS_LOG(WARNING) << "crl is nullptr. return true.";
      result = true;
      break;
    }
    const ASN1_TIME *lastUpdate = X509_CRL_get0_lastUpdate(crl);
    const ASN1_TIME *nextUpdate = X509_CRL_get0_nextUpdate(crl);

    int day = 0;
    int sec = 0;
    int ret = ASN1_TIME_diff(&day, &sec, lastUpdate, NULL);
    if (ret != 1) {
      result = false;
      break;
    }

    if (day < 0 || sec < 0) {
      MS_LOG(ERROR) << "crl start time is later than now time.";
      result = false;
      break;
    }
    day = 0;
    sec = 0;
    ret = ASN1_TIME_diff(&day, &sec, NULL, nextUpdate);
    if (ret != 1) {
      result = false;
      break;
    }

    if (day < 0 || sec < 0) {
      MS_LOG(WARNING) << "crl update time is sooner than now time. please update crl";
    }
    MS_LOG(INFO) << "verifyCRL time success.";
  } while (0);

  X509_CRL_free(crl);
  BIO_free_all(bio);
  return result;
}

std::string CommUtil::BoolToString(bool alive) {
  if (alive) {
    return "True";
  } else {
    return "False";
  }
}

bool CommUtil::StringToBool(const std::string &alive) {
  if (alive == "True") {
    return true;
  } else if (alive == "False") {
    return false;
  }
  return false;
}

Time CommUtil::GetNowTime() {
  ps::core::Time time;
  auto time_now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(time_now);
  struct tm ptm;
  (void)localtime_r(&tt, &ptm);
  std::ostringstream time_mill_oss;
  time_mill_oss << std::put_time(&ptm, "%Y-%m-%d %H:%M:%S");

  // calculate millisecond, the format of time_str_mill is 2022-01-10 20:22:20.067
  auto second_time_stamp = std::chrono::duration_cast<std::chrono::seconds>(time_now.time_since_epoch());
  auto mill_time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(time_now.time_since_epoch());
  auto ms_stamp = mill_time_stamp - second_time_stamp;
  time_mill_oss << "." << std::setfill('0') << std::setw(kMillSecondLength) << ms_stamp.count();

  time.time_stamp = LongToSize(mill_time_stamp.count());
  time.time_str_mill = time_mill_oss.str();
  return time;
}

bool CommUtil::ParseAndCheckConfigJson(Configuration *file_configuration, const std::string &key,
                                       FileConfig *file_config) {
  MS_EXCEPTION_IF_NULL(file_configuration);
  MS_EXCEPTION_IF_NULL(file_config);
  if (!file_configuration->Exists(key)) {
    MS_LOG(WARNING) << key << " config is not set. Don't write.";
    return false;
  } else {
    std::string value = file_configuration->Get(key, "");
    nlohmann::json value_json;
    try {
      value_json = nlohmann::json::parse(value);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "The hyper-parameter data is not in json format.";
    }
    // Parse the storage type.
    uint32_t storage_type = ps::core::CommUtil::JsonGetKeyWithException<uint32_t>(value_json, ps::kStoreType);
    if (std::to_string(storage_type) != ps::kFileStorage) {
      MS_LOG(EXCEPTION) << "Storage type " << storage_type << " is not supported.";
    }
    // Parse storage file path.
    std::string file_path = ps::core::CommUtil::JsonGetKeyWithException<std::string>(value_json, ps::kStoreFilePath);
    file_config->storage_type = storage_type;
    file_config->storage_file_path = file_path;
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
