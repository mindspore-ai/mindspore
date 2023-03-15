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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_WRAPPER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_WRAPPER_H_

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <assert.h>
#include <openssl/pkcs12.h>
#include <openssl/bio.h>

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "utils/log_adapter.h"
#include "ps/core/comm_util.h"
#include "ps/core/file_configuration.h"
#include "include/backend/distributed/ps/constants.h"

namespace mindspore {
namespace ps {
namespace core {
class SSLWrapper {
 public:
  void InitSSL();
  static SSLWrapper &GetInstance() {
    static SSLWrapper instance;
    return instance;
  }
  SSL_CTX *GetSSLCtx(bool is_server = true);

 private:
  SSLWrapper();
  virtual ~SSLWrapper();
  SSLWrapper(const SSLWrapper &) = delete;
  SSLWrapper &operator=(const SSLWrapper &) = delete;

  void CleanSSL();
  time_t ConvertAsn1Time(const ASN1_TIME *const time) const;
  void StartCheckCertTime(const Configuration &config, const X509 *cert, const std::string &ca_path);
  void StopCheckCertTime();
  void InitSSLCtx(const Configuration &config, const X509 *cert, const EVP_PKEY *pkey, X509_CRL *crl);

  SSL_CTX *ssl_ctx_;

  // The firset root ca certificate.
  X509 *rootFirstCA_;
  // The second root ca certificate.
  X509 *rootSecondCA_;
  std::unique_ptr<std::thread> check_time_thread_;
  std::atomic<bool> running_;
  std::atomic<bool> is_ready_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::mutex verify_mutex_;

  // Indicates whether the ssl wrapper has been initialized.
  bool init_{false};
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_WRAPPER_H_
