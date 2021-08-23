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

#include "utils/log_adapter.h"
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
class SSLWrapper {
 public:
  static SSLWrapper &GetInstance() {
    static SSLWrapper instance;
    return instance;
  }
  SSL_CTX *GetSSLCtx(bool is_server = true);

  // read certificate from file path
  X509 *ReadCertFromFile(const std::string &certPath) const;

  // read Certificate Revocation List from file absolute path
  X509_CRL *ReadCrlFromFile(const std::string &crlPath) const;

  // read certificate from pem string
  X509 *ReadCertFromPerm(std::string cert);

  // verify valid of certificate time
  bool VerifyCertTime(const X509 *cert) const;

  // verify valid of certificate chain
  bool VerifyCAChain(const std::string &keyAttestation, const std::string &equipCert, const std::string &equipCACert,
                     std::string rootCert);

  // verify valid of sign data
  bool VerifyRSAKey(const std::string &keyAttestation, const unsigned char *srcData, const unsigned char *signData,
                    int srcDataLen);

  // verify valid of equip certificate with CRL
  bool VerifyCRL(const std::string &equipCert);

 private:
  SSLWrapper();
  virtual ~SSLWrapper();
  SSLWrapper(const SSLWrapper &) = delete;
  SSLWrapper &operator=(const SSLWrapper &) = delete;

  void InitSSL();
  void CleanSSL();

  SSL_CTX *ssl_ctx_;
  SSL_CTX *client_ssl_ctx_;

  // The firset root ca certificate.
  X509 *rootFirstCA_;
  // The second root ca certificate.
  X509 *rootSecondCA_;
  // The firset root revocation list
  X509_CRL *rootFirstCrl_;
  // The second root revocation list
  X509_CRL *rootSecondCrl_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_WRAPPER_H_
