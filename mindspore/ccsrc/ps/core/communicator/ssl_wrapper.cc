
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

#include "ps/core/communicator/ssl_wrapper.h"

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
      client_ssl_ctx_(nullptr),
      rootFirstCA_(nullptr),
      rootSecondCA_(nullptr),
      rootFirstCrl_(nullptr),
      rootSecondCrl_(nullptr) {
  InitSSL();
}

SSLWrapper::~SSLWrapper() { CleanSSL(); }

void SSLWrapper::InitSSL() {
  SSL_library_init();
  ERR_load_crypto_strings();
  SSL_load_error_strings();
  OpenSSL_add_all_algorithms();
  int rand = RAND_poll();
  if (rand == 0) {
    MS_LOG(ERROR) << "RAND_poll failed";
  }
  ssl_ctx_ = SSL_CTX_new(SSLv23_server_method());
  if (!ssl_ctx_) {
    MS_LOG(ERROR) << "SSL_CTX_new failed";
  }
  X509_STORE *store = SSL_CTX_get_cert_store(ssl_ctx_);
  if (X509_STORE_set_default_paths(store) != 1) {
    MS_LOG(ERROR) << "X509_STORE_set_default_paths failed";
  }
  client_ssl_ctx_ = SSL_CTX_new(SSLv23_client_method());
  if (!ssl_ctx_) {
    MS_LOG(ERROR) << "SSL_CTX_new failed";
  }
}

void SSLWrapper::CleanSSL() {
  if (ssl_ctx_ != nullptr) {
    SSL_CTX_free(ssl_ctx_);
  }
  ERR_free_strings();
  EVP_cleanup();
  ERR_remove_thread_state(nullptr);
  CRYPTO_cleanup_all_ex_data();
}

SSL_CTX *SSLWrapper::GetSSLCtx(bool is_server) {
  if (is_server) {
    return ssl_ctx_;
  } else {
    return client_ssl_ctx_;
  }
}

X509 *SSLWrapper::ReadCertFromFile(const std::string &certPath) const {
  BIO *bio = BIO_new_file(certPath.c_str(), "r");
  return PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
}

X509 *SSLWrapper::ReadCertFromPerm(std::string cert) {
  BIO *bio = BIO_new_mem_buf(reinterpret_cast<void *>(cert.data()), -1);
  return PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
}

X509_CRL *SSLWrapper::ReadCrlFromFile(const std::string &crlPath) const {
  BIO *bio = BIO_new_file(crlPath.c_str(), "r");
  return PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr);
}

bool SSLWrapper::VerifyCertTime(const X509 *cert) const {
  ASN1_TIME *start = X509_getm_notBefore(cert);
  ASN1_TIME *end = X509_getm_notAfter(cert);

  int day = 0;
  int sec = 0;
  ASN1_TIME_diff(&day, &sec, start, NULL);

  if (day < 0 || sec < 0) {
    MS_LOG(INFO) << "Cert start time is later than now time.";
    return false;
  }
  day = 0;
  sec = 0;
  ASN1_TIME_diff(&day, &sec, NULL, end);
  if (day < 0 || sec < 0) {
    MS_LOG(INFO) << "Cert end time is sooner than now time.";
    return false;
  }

  return true;
}

bool SSLWrapper::VerifyCAChain(const std::string &keyAttestation, const std::string &equipCert,
                               const std::string &equipCACert, std::string) {
  X509 *keyAttestationCertObj = ReadCertFromPerm(keyAttestation);
  X509 *equipCertObj = ReadCertFromPerm(equipCert);
  X509 *equipCACertObj = ReadCertFromPerm(equipCACert);

  if (!VerifyCertTime(keyAttestationCertObj) || !VerifyCertTime(equipCertObj) || !VerifyCertTime(equipCACertObj)) {
    return false;
  }

  EVP_PKEY *equipPubKey = X509_get_pubkey(equipCertObj);
  EVP_PKEY *equipCAPubKey = X509_get_pubkey(equipCACertObj);

  EVP_PKEY *rootFirstPubKey = X509_get_pubkey(rootFirstCA_);
  EVP_PKEY *rootSecondPubKey = X509_get_pubkey(rootSecondCA_);

  int ret = 0;
  ret = X509_verify(keyAttestationCertObj, equipPubKey);
  if (ret != 1) {
    MS_LOG(INFO) << "keyAttestationCert verify is failed";
    return false;
  }
  ret = X509_verify(equipCertObj, equipCAPubKey);
  if (ret != 1) {
    MS_LOG(INFO) << "Equip cert verify is failed";
    return false;
  }
  int ret_first = X509_verify(equipCACertObj, rootFirstPubKey);
  int ret_second = X509_verify(equipCACertObj, rootSecondPubKey);
  if (ret_first != 1 && ret_second != 1) {
    MS_LOG(INFO) << "Equip ca cert verify is failed";
    return false;
  }
  MS_LOG(INFO) << "VerifyCAChain success.";

  EVP_PKEY_free(equipPubKey);
  EVP_PKEY_free(equipCAPubKey);
  EVP_PKEY_free(rootFirstPubKey);
  EVP_PKEY_free(rootSecondPubKey);
  return true;
}

bool SSLWrapper::VerifyCRL(const std::string &equipCert) {
  X509 *equipCertObj = ReadCertFromPerm(equipCert);
  if (rootFirstCrl_ == nullptr && rootSecondCrl_ == nullptr) {
    MS_LOG(INFO) << "RootFirstCrl && rootSecondCrl is nullptr.";
    return false;
  }

  EVP_PKEY *evp_pkey = X509_get_pubkey(equipCertObj);
  int ret = X509_CRL_verify(rootFirstCrl_, evp_pkey);
  if (ret == 1) {
    MS_LOG(INFO) << "Equip cert in root first crl, verify failed";
    return false;
  }
  ret = X509_CRL_verify(rootSecondCrl_, evp_pkey);
  if (ret == 1) {
    MS_LOG(INFO) << "Equip cert in root second crl, verify failed";
    return false;
  }
  MS_LOG(INFO) << "VerifyCRL success.";
  return true;
}

bool SSLWrapper::VerifyRSAKey(const std::string &keyAttestation, const unsigned char *srcData,
                              const unsigned char *signData, int srcDataLen) {
  if (keyAttestation.empty() || srcData == nullptr || signData == nullptr) {
    MS_LOG(INFO) << "KeyAttestation or srcData or signData is empty.";
    return false;
  }

  X509 *keyAttestationCertObj = ReadCertFromPerm(keyAttestation);

  EVP_PKEY *pubKey = X509_get_pubkey(keyAttestationCertObj);
  RSA *pRSAPublicKey = EVP_PKEY_get0_RSA(pubKey);
  if (pRSAPublicKey == nullptr) {
    MS_LOG(INFO) << "Get rsa public key failed.";
    return false;
  }

  int pubKeyLen = RSA_size(pRSAPublicKey);
  int ret = RSA_verify(NID_sha256, srcData, srcDataLen, signData, pubKeyLen, pRSAPublicKey);
  if (ret != 1) {
    MS_LOG(WARNING) << "Verify error.";
    int64_t ulErr = ERR_get_error();
    char szErrMsg[1024] = {0};
    MS_LOG(WARNING) << "Error number: " << ulErr;
    ERR_error_string(ulErr, szErrMsg);
    MS_LOG(INFO) << "Error message:" << szErrMsg;
    return false;
  }
  RSA_free(pRSAPublicKey);
  X509_free(keyAttestationCertObj);
  CRYPTO_cleanup_all_ex_data();

  MS_LOG(INFO) << "VerifyRSAKey success.";
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
