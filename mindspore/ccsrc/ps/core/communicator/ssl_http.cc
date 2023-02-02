
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

#include "ps/core/communicator/ssl_http.h"

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
SSLHTTP::SSLHTTP() : ssl_ctx_(nullptr) { InitSSL(); }

SSLHTTP::~SSLHTTP() { CleanSSL(); }

void SSLHTTP::InitSSL() {
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
  ssl_ctx_ = SSL_CTX_new(SSLv23_server_method());
  if (!ssl_ctx_) {
    MS_LOG(EXCEPTION) << "SSL_CTX_new failed";
  }
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
  std::string server_cert = kCertificateChain;
  std::string path = CommUtil::ParseConfig(*(config_), kServerCertPath);
  if (!CommUtil::IsFileExists(path)) {
    MS_LOG(EXCEPTION) << "The key:" << kServerCertPath << "'s value is not exist.";
  }
  server_cert = path;

  // 2. Parse the server password.
  EVP_PKEY *pkey = nullptr;
  X509 *cert = nullptr;
  STACK_OF(X509) *ca_stack = nullptr;
  BIO *bio = BIO_new_file(server_cert.c_str(), "rb");
  if (bio == nullptr) {
    MS_LOG(EXCEPTION) << "Read server cert file failed.";
  }
  PKCS12 *p12 = d2i_PKCS12_bio(bio, nullptr);
  if (p12 == nullptr) {
    MS_LOG(EXCEPTION) << "Create PKCS12 cert failed, please check whether the certificate is correct.";
  }
  BIO_free_all(bio);
  if (!PKCS12_parse(p12, PSContext::instance()->server_password(), &pkey, &cert, &ca_stack)) {
    MS_LOG(EXCEPTION) << "PKCS12_parse failed.";
  }
  PKCS12_free(p12);
  if (cert == nullptr) {
    MS_LOG(EXCEPTION) << "the cert is nullptr";
  }
  if (pkey == nullptr) {
    MS_LOG(EXCEPTION) << "the key is nullptr";
  }
  if (!CommUtil::verifyCertTimeStamp(cert)) {
    MS_LOG(EXCEPTION) << "Verify Cert Time failed.";
  }
  std::string default_cipher_list = CommUtil::ParseConfig(*config_, kCipherList);
  InitSSLCtx(cert, pkey, default_cipher_list);
  EVP_PKEY_free(pkey);
  X509_free(cert);
}

void SSLHTTP::InitSSLCtx(const X509 *cert, const EVP_PKEY *pkey, const std::string &default_cipher_list) {
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
  SSL_CTX_set_security_level(ssl_ctx_, kSecurityLevel);
}

void SSLHTTP::CleanSSL() {
  if (ssl_ctx_ != nullptr) {
    SSL_CTX_free(ssl_ctx_);
  }
  ERR_free_strings();
  EVP_cleanup();
  ERR_remove_thread_state(nullptr);
  CRYPTO_cleanup_all_ex_data();
}

SSL_CTX *SSLHTTP::GetSSLCtx() const { return ssl_ctx_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
