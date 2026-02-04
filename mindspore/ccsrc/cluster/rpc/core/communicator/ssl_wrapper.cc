
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

#include "cluster/rpc/core/communicator/ssl_wrapper.h"

#include <sys/time.h>
#include <openssl/pem.h>
#include <openssl/sha.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <sstream>

namespace mindspore {
namespace ps {
namespace core {
SSLWrapper::SSLWrapper()
    : ssl_ctx_(nullptr),
      rootFirstCA_(nullptr),
      rootSecondCA_(nullptr),
      check_time_thread_(nullptr),
      running_(false),
      is_ready_(false) {}

SSLWrapper::~SSLWrapper() { CleanSSL(); }

void SSLWrapper::InitSSL() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (init_) {
    return;
  }
  init_ = true;
  CommUtil::InitOpensslLib();
  ssl_ctx_ = SSL_CTX_new(TLS_server_method());
  if (!ssl_ctx_) {
    MS_LOG(EXCEPTION) << "SSL_CTX_new failed";
  }
  SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);
  X509_STORE *store = SSL_CTX_get_cert_store(ssl_ctx_);
  MS_EXCEPTION_IF_NULL(store);
  if (X509_STORE_set_default_paths(store) != 1) {
    MS_LOG(EXCEPTION) << "X509_STORE_set_default_paths failed";
  }
  std::unique_ptr<Configuration> config_ =
    std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(EXCEPTION) << "The config file is empty.";
  }

  // 1.Parse the server's certificate and the ciphertext of key.
  std::string path = CommUtil::ParseConfig(*(config_), kServerCertPath);
  if (!CommUtil::IsFileExists(path)) {
    MS_LOG(EXCEPTION) << "The key:" << kServerCertPath << "'s value is not exist.";
  }
  std::string server_cert = path;
  MS_LOG(INFO) << "The server cert path:" << server_cert;

  // 2. Parse the server password.
  char *server_password = PSContext::instance()->server_password();
  if (strlen(server_password) == 0) {
    MS_LOG(EXCEPTION) << "The client password's value is empty.";
  }

  EVP_PKEY *pkey = nullptr;
  X509 *cert = nullptr;
  STACK_OF(X509) *ca_stack = nullptr;
  BIO *bio = BIO_new_file(server_cert.c_str(), "rb");
  if (bio == nullptr) {
    PSContext::instance()->ClearServerPassword();
    MS_LOG(EXCEPTION) << "Read server cert file failed.";
  }
  PKCS12 *p12 = d2i_PKCS12_bio(bio, nullptr);
  BIO_free_all(bio);
  if (p12 == nullptr) {
    PSContext::instance()->ClearServerPassword();
    MS_LOG(EXCEPTION) << "Create PKCS12 cert failed, please check whether the certificate is correct.";
  }
  if (PKCS12_parse(p12, server_password, &pkey, &cert, &ca_stack) == 0) {
    if (ERR_GET_REASON(ERR_peek_last_error()) == PKCS12_R_MAC_VERIFY_FAILURE) {
      PSContext::instance()->ClearServerPassword();
      MS_LOG(EXCEPTION) << "The server password is invalid!";
    }
    PSContext::instance()->ClearServerPassword();
    MS_LOG(EXCEPTION) << "PKCS12_parse failed, the reason is " << ERR_reason_error_string(ERR_peek_last_error());
  }
  PSContext::instance()->ClearServerPassword();

  PKCS12_free(p12);
  MS_EXCEPTION_IF_NULL(cert);
  MS_EXCEPTION_IF_NULL(pkey);
  if (ca_stack != nullptr) {
    MS_LOG(EXCEPTION) << "The cert is invalid: ca_stack should be empty.";
  }

  std::string ca_path = CommUtil::ParseConfig(*config_, kCaCertPath);
  if (!CommUtil::IsFileExists(ca_path)) {
    MS_LOG(EXCEPTION) << "The key:" << kCaCertPath << "'s value is not exist.";
  }
  BIO *ca_bio = BIO_new_file(ca_path.c_str(), "r");
  if (ca_bio == nullptr) {
    MS_LOG(EXCEPTION) << "Read CA cert file failed.";
  }
  X509 *caCert = PEM_read_bio_X509(ca_bio, nullptr, nullptr, nullptr);
  X509_CRL *crl = nullptr;
  std::string crl_path = CommUtil::ParseConfig(*(config_), kCrlPath);
  if (crl_path.empty()) {
    MS_LOG(INFO) << "The crl path is empty.";
  } else if (!CommUtil::checkCRLTime(crl_path)) {
    MS_LOG(EXCEPTION) << "check crl time failed";
  } else if (!CommUtil::VerifyCRL(caCert, crl_path, &crl)) {
    MS_LOG(EXCEPTION) << "Verify crl failed.";
  }

  CommUtil::verifyCertPipeline(caCert, cert);

  SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, 0);
  if (!SSL_CTX_load_verify_locations(ssl_ctx_, ca_path.c_str(), nullptr)) {
    MS_LOG(EXCEPTION) << "SSL load ca location failed!";
  }

  InitSSLCtx(*config_, cert, pkey, crl);
  StartCheckCertTime(*config_, cert, ca_path);

  EVP_PKEY_free(pkey);
  X509_free(caCert);
  X509_free(cert);
  BIO_vfree(ca_bio);
  if (crl != nullptr) {
    X509_CRL_free(crl);
  }
}

void SSLWrapper::InitSSLCtx(const Configuration &config, const X509 *cert, const EVP_PKEY *pkey, X509_CRL *crl) {
  std::string default_cipher_list = CommUtil::ParseConfig(config, kCipherList);
  std::vector<std::string> ciphers = CommUtil::Split(default_cipher_list, kColon);
  if (!CommUtil::VerifyCipherList(ciphers)) {
    MS_LOG(EXCEPTION) << "The cipher is wrong.";
  }
  if (!SSL_CTX_set_cipher_list(ssl_ctx_, default_cipher_list.c_str())) {
    MS_LOG(EXCEPTION) << "SSL use set cipher list failed!";
  }
  if (!SSL_CTX_use_certificate(ssl_ctx_, const_cast<X509 *>(cert))) {
    MS_LOG(EXCEPTION) << "SSL use certificate chain file failed!";
  }

  if (!SSL_CTX_use_PrivateKey(ssl_ctx_, const_cast<EVP_PKEY *>(pkey))) {
    MS_LOG(EXCEPTION) << "SSL use private key file failed!";
  }

  if (!SSL_CTX_check_private_key(ssl_ctx_)) {
    MS_LOG(EXCEPTION) << "SSL check private key file failed!";
  }
  if (!SSL_CTX_set_options(ssl_ctx_, SSL_OP_SINGLE_DH_USE | SSL_OP_SINGLE_ECDH_USE | SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 |
                                       SSL_OP_NO_TLSv1 | SSL_OP_NO_TLSv1_1)) {
    MS_LOG(EXCEPTION) << "SSL_CTX_set_options failed.";
  }
  if (!SSL_CTX_set_mode(ssl_ctx_, SSL_MODE_AUTO_RETRY)) {
    MS_LOG(EXCEPTION) << "SSL set mode auto retry failed!";
  }

  if (crl != nullptr) {
    // Load CRL into the `X509_STORE`
    X509_STORE *x509_store = SSL_CTX_get_cert_store(ssl_ctx_);
    if (X509_STORE_add_crl(x509_store, crl) != 1) {
      MS_LOG(EXCEPTION) << "ssl server X509_STORE add crl failed!";
    }

    // Enable CRL checking
    X509_VERIFY_PARAM *param = SSL_CTX_get0_param(ssl_ctx_);
    if (param == nullptr) {
      MS_LOG(EXCEPTION) << "ssl server X509_VERIFY_PARAM is nullptr!";
    }
    if (X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK) != 1) {
      MS_LOG(EXCEPTION) << "ssl server X509_VERIFY_PARAM set flag X509_V_FLAG_CRL_CHECK failed!";
    }
  }

  SSL_CTX_set_security_level(ssl_ctx_, kSecurityLevel);
}

void SSLWrapper::CleanSSL() {
  if (ssl_ctx_ != nullptr) {
    SSL_CTX_free(ssl_ctx_);
  }
  ERR_free_strings();
  EVP_cleanup();
  ERR_remove_thread_state(nullptr);
  CRYPTO_cleanup_all_ex_data();
  StopCheckCertTime();
}

void SSLWrapper::StartCheckCertTime(const Configuration &config, const X509 *cert, const std::string &ca_path) {
  MS_EXCEPTION_IF_NULL(cert);
  MS_LOG(INFO) << "The server start check cert.";
  int64_t interval = kCertCheckIntervalInHour;

  int64_t warning_time = kCertExpireWarningTimeInDay;
  if (config.Exists(kCertExpireWarningTime)) {
    int64_t res_time = config.GetInt(kCertExpireWarningTime, 0);
    if (res_time < kMinWarningTime || res_time > kMaxWarningTime) {
      MS_LOG(EXCEPTION) << "The Certificate expiration warning time should be [7, 180]";
    }
    warning_time = res_time;
  }
  MS_LOG(INFO) << "The interval time is:" << interval << ", the warning time is:" << warning_time;
  BIO *ca_bio = BIO_new_file(ca_path.c_str(), "r");
  MS_EXCEPTION_IF_NULL(ca_bio);
  X509 *ca_cert = PEM_read_bio_X509(ca_bio, nullptr, nullptr, nullptr);
  BIO_free_all(ca_bio);
  MS_EXCEPTION_IF_NULL(ca_cert);

  running_ = true;
  check_time_thread_ = std::make_unique<std::thread>([&, cert, ca_cert, interval, warning_time]() {
    while (running_) {
      if (!CommUtil::VerifyCertTime(cert, warning_time)) {
        MS_LOG(WARNING) << "Verify server cert time failed.";
      }

      if (!CommUtil::VerifyCertTime(ca_cert, warning_time)) {
        MS_LOG(WARNING) << "Verify ca cert time failed.";
      }
      std::unique_lock<std::mutex> lock(mutex_);
      bool res = cond_.wait_for(lock, std::chrono::hours(interval), [&] {
        bool result = is_ready_.load();
        return result;
      });
      MS_LOG(INFO) << "Wait for res:" << res;
    }
  });
  MS_EXCEPTION_IF_NULL(check_time_thread_);
}

void SSLWrapper::StopCheckCertTime() {
  running_ = false;
  is_ready_ = true;
  cond_.notify_all();
  if (check_time_thread_ != nullptr) {
    check_time_thread_->join();
  }
}

SSL_CTX *SSLWrapper::GetSSLCtx(bool) { return ssl_ctx_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
