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

#ifndef MINDSPORE_CCSRC_FL_SERVER_CERT_VERIFY_H
#define MINDSPORE_CCSRC_FL_SERVER_CERT_VERIFY_H

#include <assert.h>
#ifndef _WIN32
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/x509v3.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#endif
#include <iostream>
#include <fstream>
#include <string>
#include "utils/log_adapter.h"
#include "fl/server/common.h"

namespace mindspore {
namespace ps {
namespace server {
class CertVerify {
 public:
  CertVerify() {}
  ~CertVerify() = default;

  bool verifyCertAndSign(const std::string &flID, const std::string &timeStamp, const unsigned char *signData,
                         const std::string &keyAttestation, const std::string &equipCert,
                         const std::string &equipCACert, const std::string &rootFirstCAPath,
                         const std::string &rootSecondCAPath, const std::string &equipCrlPath);

  static bool initRootCertAndCRL(const std::string rootFirstCaFilePath, const std::string rootSecondCaFilePath,
                                 const std::string equipCrlPath, uint64_t replay_attack_time_diff_);

  // verify valid of sign data
  bool verifyRSAKey(const std::string &keyAttestation, const uint8_t *srcData, const uint8_t *signData, int srcDataLen);

  void sha256Hash(const uint8_t *src, const int src_len, uint8_t *hash, const int len) const;

  // verify valid of time stamp of request
  bool verifyTimeStamp(const std::string &flID, const std::string &timeStamp) const;

#ifndef _WIN32

 private:
  // read certificate from file path
  static X509 *readCertFromFile(const std::string &certPath);

  // read Certificate Revocation List from file absolute path
  static X509_CRL *readCrlFromFile(const std::string &crlPath);

  // read certificate from pem string
  X509 *readCertFromPerm(std::string cert);

  // verify valid of certificate time
  bool verifyCertTime(const X509 *cert) const;

  // verify valid of certificate chain
  bool verifyCAChain(const std::string &keyAttestation, const std::string &equipCert, const std::string &equipCACert,
                     const std::string &rootFirstCAPath, const std::string &rootSecondCAPath);

  // verify valid of sign data
  bool verifyRSAKey(const std::string &keyAttestation, const unsigned char *signData, const std::string &flID,
                    const std::string &timeStamp);

  // verify valid of equip certificate with CRL
  bool verifyCRL(const std::string &equipCert, const std::string &equipCrlPath);

  // verify valid of flID with sha256(equip cert)
  bool verifyEquipCertAndFlID(const std::string &flID, const std::string &equipCert);

  void sha256Hash(const std::string &src, uint8_t *hash, const int len) const;

  std::string toHexString(const unsigned char *data, const int len);

  bool verifyCertCommonName(const X509 *caCert, const X509 *subCert) const;

  bool verifyExtendedAttributes(const X509 *cert) const;

  bool verifyCertKeyID(const X509 *caCert, const X509 *subCert) const;

  bool verifyPublicKey(const X509 *keyAttestationCertObj, const X509 *equipCertObj, const X509 *equipCACertObj,
                       const X509 *rootFirstCA, const X509 *rootSecondCA) const;
#endif
};
}  // namespace server
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_CERT_VERIFY_H
