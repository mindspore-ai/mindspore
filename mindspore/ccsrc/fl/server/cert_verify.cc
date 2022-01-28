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
#include "fl/server/cert_verify.h"
#include <sys/time.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <sstream>

namespace mindspore {
namespace ps {
namespace server {
#ifndef _WIN32
static const int64_t certStartTimeDiff = -600;
static int64_t replayAttackTimeDiff;

X509 *CertVerify::readCertFromFile(const std::string &certPath) {
  BIO *bio = BIO_new_file(certPath.c_str(), "r");
  X509 *certObj = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
  BIO_free_all(bio);
  return certObj;
}

X509 *CertVerify::readCertFromPerm(std::string cert) {
  BIO *bio = BIO_new_mem_buf(reinterpret_cast<void *>(cert.data()), -1);
  X509 *certObj = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
  BIO_free_all(bio);
  return certObj;
}

X509_CRL *CertVerify::readCrlFromFile(const std::string &crlPath) {
  BIO *bio = BIO_new_file(crlPath.c_str(), "r");
  X509_CRL *crl = PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr);
  BIO_free_all(bio);
  return crl;
}

bool checkFileExists(const std::string &file) {
  std::ifstream f(file.c_str());
  if (!f.good()) {
    return false;
  } else {
    f.close();
    return true;
  }
}

bool CertVerify::verifyCertTime(const X509 *cert) const {
  ASN1_TIME *start = X509_getm_notBefore(cert);
  ASN1_TIME *end = X509_getm_notAfter(cert);

  int day = 0;
  int sec = 0;
  int ret = ASN1_TIME_diff(&day, &sec, start, NULL);
  if (ret != 1) {
    return false;
  }
  if (day < 0) {
    MS_LOG(WARNING) << "cert start day time is later than now day time, day is" << day;
    return false;
  }
  if (day == 0 && sec < certStartTimeDiff) {
    MS_LOG(WARNING) << "cert start second time is later than 600 second, second is" << sec;
    return false;
  }
  day = 0;
  sec = 0;
  ret = ASN1_TIME_diff(&day, &sec, NULL, end);
  if (ret != 1) {
    return false;
  }

  if (day < 0 || sec < 0) {
    MS_LOG(WARNING) << "cert end time is sooner than now time.";
    return false;
  }
  MS_LOG(DEBUG) << "verify cert time end.";
  return true;
}

bool CertVerify::verifyPublicKey(const X509 *keyAttestationCertObj, const X509 *equipCertObj,
                                 const X509 *equipCACertObj, const X509 *rootFirstCA, const X509 *rootSecondCA) const {
  bool result = true;
  EVP_PKEY *equipPubKey = X509_get_pubkey(const_cast<X509 *>(equipCertObj));
  EVP_PKEY *equipCAPubKey = X509_get_pubkey(const_cast<X509 *>(equipCACertObj));
  EVP_PKEY *rootFirstPubKey = X509_get_pubkey(const_cast<X509 *>(rootFirstCA));
  EVP_PKEY *rootSecondPubKey = X509_get_pubkey(const_cast<X509 *>(rootSecondCA));
  do {
    int ret = 0;
    ret = X509_verify(const_cast<X509 *>(keyAttestationCertObj), equipPubKey);
    if (ret != 1) {
      MS_LOG(WARNING) << "keyAttestationCert verify is failed";
      result = false;
      break;
    }
    ret = X509_verify(const_cast<X509 *>(equipCertObj), equipCAPubKey);
    if (ret != 1) {
      MS_LOG(WARNING) << "equip cert verify is failed";
      result = false;
      break;
    }
    int ret_first = X509_verify(const_cast<X509 *>(equipCACertObj), rootFirstPubKey);
    int ret_second = X509_verify(const_cast<X509 *>(equipCACertObj), rootSecondPubKey);
    if (ret_first != 1 && ret_second != 1) {
      MS_LOG(WARNING) << "equip ca cert verify is failed";
      result = false;
      break;
    }
  } while (0);

  EVP_PKEY_free(equipPubKey);
  EVP_PKEY_free(equipCAPubKey);
  EVP_PKEY_free(rootFirstPubKey);
  EVP_PKEY_free(rootSecondPubKey);
  MS_LOG(DEBUG) << "verify Public Key end.";
  return result;
}

bool CertVerify::verifyCAChain(const std::string &keyAttestation, const std::string &equipCert,
                               const std::string &equipCACert, const std::string &rootFirstCAPath,
                               const std::string &rootSecondCAPath) {
  X509 *rootFirstCA = CertVerify::readCertFromFile(rootFirstCAPath);
  X509 *rootSecondCA = CertVerify::readCertFromFile(rootSecondCAPath);
  X509 *keyAttestationCertObj = readCertFromPerm(keyAttestation);
  X509 *equipCertObj = readCertFromPerm(equipCert);
  X509 *equipCACertObj = readCertFromPerm(equipCACert);
  bool result = true;
  do {
    if (rootFirstCA == nullptr || rootSecondCA == nullptr) {
      MS_LOG(WARNING) << "rootFirstCA or rootSecondCA is nullptr";
      result = false;
      break;
    }
    if (keyAttestationCertObj == nullptr || equipCertObj == nullptr || equipCACertObj == nullptr) {
      result = false;
      break;
    }

    if (!verifyCertTime(keyAttestationCertObj) || !verifyCertTime(equipCertObj) || !verifyCertTime(equipCACertObj)) {
      result = false;
      break;
    }

    if (!verifyCertCommonName(equipCACertObj, equipCertObj)) {
      MS_LOG(WARNING) << "equip ca cert subject cn is not equal with equip cert issuer cn.";
      result = false;
      break;
    }

    if (!verifyCertCommonName(rootFirstCA, equipCACertObj) && !verifyCertCommonName(rootSecondCA, equipCACertObj)) {
      MS_LOG(WARNING) << "root CA cert subject cn is not equal with equip CA cert issuer cn.";
      result = false;
      break;
    }

    if (!verifyExtendedAttributes(equipCACertObj)) {
      MS_LOG(WARNING) << "verify equipCACert Extended Attributes failed.";
      result = false;
      break;
    }

    if (!verifyCertKeyID(rootFirstCA, equipCACertObj) && !verifyCertKeyID(rootSecondCA, equipCACertObj)) {
      MS_LOG(WARNING) << "root CA cert subject keyid is not equal with equip CA cert issuer keyid.";
      result = false;
      break;
    }

    if (!verifyCertKeyID(equipCACertObj, equipCertObj)) {
      MS_LOG(WARNING) << "equip CA cert subject keyid is not equal with equip cert issuer keyid.";
      result = false;
      break;
    }

    if (!verifyPublicKey(keyAttestationCertObj, equipCertObj, equipCACertObj, rootFirstCA, rootSecondCA)) {
      MS_LOG(WARNING) << "verify Public Key failed";
      result = false;
      break;
    }
  } while (0);
  X509_free(rootFirstCA);
  X509_free(rootSecondCA);
  X509_free(keyAttestationCertObj);
  X509_free(equipCertObj);
  X509_free(equipCACertObj);
  MS_LOG(DEBUG) << "verifyCAChain end.";
  return result;
}

bool CertVerify::verifyCertKeyID(const X509 *caCert, const X509 *subCert) const {
  bool result = true;
  ASN1_OCTET_STRING *skid = nullptr;
  AUTHORITY_KEYID *akeyid = nullptr;
  do {
    int crit = 0;
    skid = reinterpret_cast<ASN1_OCTET_STRING *>(X509_get_ext_d2i(caCert, NID_subject_key_identifier, &crit, NULL));
    if (skid == nullptr) {
      result = false;
      break;
    }
    char subject_keyid[512] = {0};
    for (int i = 0; i < skid->length; i++) {
      char keyid[8] = {0};
      size_t base = 512;
      (void)sprintf_s(keyid, sizeof(keyid), "%x ", (uint32_t)skid->data[i]);
      int ret = strcat_s(subject_keyid, base, keyid);
      if (ret == -1) {
        result = false;
        break;
      }
    }

    akeyid = reinterpret_cast<AUTHORITY_KEYID *>(X509_get_ext_d2i(subCert, NID_authority_key_identifier, &crit, NULL));
    if (akeyid == nullptr) {
      result = false;
      break;
    }
    char issuer_keyid[512] = {0};
    if (akeyid->keyid == nullptr) {
      MS_LOG(WARNING) << "keyid is nullprt.";
      result = false;
      break;
    }
    for (int i = 0; i < akeyid->keyid->length; i++) {
      char keyid[8] = {0};
      size_t base = 512;
      (void)sprintf_s(keyid, sizeof(keyid), "%x ", (uint32_t)(akeyid->keyid->data[i]));
      int ret = strcat_s(issuer_keyid, base, keyid);
      if (ret == -1) {
        result = false;
        break;
      }
    }

    std::string subject_keyid_str = subject_keyid;
    std::string issuer_keyid_str = issuer_keyid;
    if (subject_keyid_str != issuer_keyid_str) {
      result = false;
      break;
    }
  } while (0);
  ASN1_OCTET_STRING_free(skid);
  AUTHORITY_KEYID_free(akeyid);
  return result;
}

bool CertVerify::verifyExtendedAttributes(const X509 *cert) const {
  bool result = true;
  BASIC_CONSTRAINTS *bcons = nullptr;
  ASN1_BIT_STRING *lASN1UsageStr = nullptr;
  do {
    int cirt = 0;
    bcons = reinterpret_cast<BASIC_CONSTRAINTS *>(X509_get_ext_d2i(cert, NID_basic_constraints, &cirt, NULL));
    if (bcons == nullptr) {
      result = false;
      break;
    }
    if (!bcons->ca) {
      MS_LOG(WARNING) << "Subject Type is End Entity.";
      result = false;
      break;
    }
    MS_LOG(DEBUG) << "Subject Type is CA.";

    lASN1UsageStr = reinterpret_cast<ASN1_BIT_STRING *>(X509_get_ext_d2i(cert, NID_key_usage, NULL, NULL));
    if (lASN1UsageStr == nullptr) {
      result = false;
      break;
    }
    int16_t usage = lASN1UsageStr->data[0];
    if (lASN1UsageStr->length > 1) {
      const unsigned int move = 8;
      usage |= lASN1UsageStr->data[1] << move;
    }

    if (!(usage & KU_KEY_CERT_SIGN)) {
      MS_LOG(WARNING) << "Subject is not Certificate Signature.";
      result = false;
      break;
    }
    MS_LOG(DEBUG) << "Subject is Certificate Signature.";
  } while (0);
  BASIC_CONSTRAINTS_free(bcons);
  ASN1_BIT_STRING_free(lASN1UsageStr);
  return result;
}

bool CertVerify::verifyCertCommonName(const X509 *caCert, const X509 *subCert) const {
  if (caCert == nullptr || subCert == nullptr) {
    return false;
  }

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
    return false;
  }
  return true;
}

bool CertVerify::verifyCRL(const std::string &equipCert, const std::string &equipCrlPath) {
  if (!checkFileExists(equipCrlPath)) {
    return true;
  }
  bool result = true;
  X509_CRL *equipCrl = nullptr;
  X509 *equipCertObj = nullptr;
  EVP_PKEY *evp_pkey = nullptr;
  do {
    equipCrl = CertVerify::readCrlFromFile(equipCrlPath);
    equipCertObj = readCertFromPerm(equipCert);
    if (equipCertObj == nullptr) {
      result = false;
      break;
    }

    if (equipCrl == nullptr) {
      MS_LOG(DEBUG) << "equipCrl is nullptr. return true.";
      result = true;
      break;
    }
    evp_pkey = X509_get_pubkey(equipCertObj);
    int ret = X509_CRL_verify(equipCrl, evp_pkey);
    if (ret == 1) {
      MS_LOG(WARNING) << "equip cert in equip crl, verify failed";
      result = false;
      break;
    }
  } while (0);

  EVP_PKEY_free(evp_pkey);
  X509_free(equipCertObj);
  X509_CRL_free(equipCrl);
  MS_LOG(DEBUG) << "verifyCRL end.";
  return result;
}

bool CertVerify::verifyRSAKey(const std::string &keyAttestation, const unsigned char *signData, const std::string &flID,
                              const std::string &timeStamp) {
  if (keyAttestation.empty() || signData == nullptr || flID.empty() || timeStamp.empty()) {
    MS_LOG(WARNING) << "keyAttestation or signData or flID or timeStamp is empty.";
    return false;
  }
  bool result = true;
  X509 *keyAttestationCertObj = nullptr;
  EVP_PKEY *pubKey = nullptr;
  do {
    keyAttestationCertObj = readCertFromPerm(keyAttestation);

    std::string srcData = flID + " " + timeStamp;
    // SHA256_DIGEST_LENGTH is 32
    unsigned char srcDataHash[SHA256_DIGEST_LENGTH];
    sha256Hash(srcData, srcDataHash, SHA256_DIGEST_LENGTH);

    pubKey = X509_get_pubkey(keyAttestationCertObj);
    RSA *pRSAPublicKey = EVP_PKEY_get0_RSA(pubKey);
    if (pRSAPublicKey == nullptr) {
      MS_LOG(WARNING) << "get rsa public key failed.";
      result = false;
      break;
    }

    int pubKeyLen = RSA_size(pRSAPublicKey);
    unsigned char buffer[256];
    int ret = RSA_public_decrypt(pubKeyLen, signData, buffer, pRSAPublicKey, RSA_NO_PADDING);
    if (ret == -1) {
      MS_LOG(WARNING) << "rsa public decrypt failed.";
      result = false;
      break;
    }

    int saltLen = -2;
    ret = RSA_verify_PKCS1_PSS(pRSAPublicKey, srcDataHash, EVP_sha256(), buffer, saltLen);
    if (ret != 1) {
      uint64_t ulErr = ERR_get_error();
      char szErrMsg[1024] = {0};
      MS_LOG(WARNING) << "verify WARNING. WARNING number: " << ulErr;
      std::string str_res = ERR_error_string(ulErr, szErrMsg);
      MS_LOG(WARNING) << szErrMsg;
      if (str_res.empty()) {
        result = false;
        break;
      }
      result = false;
      break;
    }
  } while (0);
  EVP_PKEY_free(pubKey);
  X509_free(keyAttestationCertObj);
  CRYPTO_cleanup_all_ex_data();

  MS_LOG(DEBUG) << "verifyRSAKey end.";
  return result;
}

void CertVerify::sha256Hash(const uint8_t *src, const int src_len, uint8_t *hash, const int len) const {
  if (len <= 0) {
    return;
  }
  SHA256_CTX sha_ctx;
  int ret = SHA256_Init(&sha_ctx);
  if (ret != 1) {
    return;
  }
  ret = SHA256_Update(&sha_ctx, src, src_len);
  if (ret != 1) {
    return;
  }
  ret = SHA256_Final(hash, &sha_ctx);
  if (ret != 1) {
    return;
  }
}

std::string CertVerify::toHexString(const unsigned char *data, const int len) {
  if (data == nullptr) {
    MS_LOG(WARNING) << "data hash is null.";
    return "";
  }

  if (len <= 0) {
    return "";
  }
  std::stringstream ss;
  int base = 2;
  for (int i = 0; i < len; i++) {
    ss << std::hex << std::setw(base) << std::setfill('0') << static_cast<int>(data[i]);
  }
  return ss.str();
}

bool CertVerify::verifyEquipCertAndFlID(const std::string &flID, const std::string &equipCert) {
  unsigned char hash[SHA256_DIGEST_LENGTH] = {""};
  sha256Hash(equipCert, hash, SHA256_DIGEST_LENGTH);
  std::string equipCertSha256 = toHexString(hash, SHA256_DIGEST_LENGTH);
  if (flID == equipCertSha256) {
    MS_LOG(DEBUG) << "verifyEquipCertAndFlID success.";
    return true;
  } else {
    MS_LOG(WARNING) << "verifyEquipCertAndFlID failed.";
    return false;
  }
}

bool CertVerify::verifyTimeStamp(const std::string &flID, const std::string &timeStamp) const {
  int64_t requestTime = std::stoll(timeStamp.c_str());
  const int64_t base = 1000;
  struct timeval tv {};
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    return false;
  }
  int64_t now = tv.tv_sec * base + tv.tv_usec / base;
  MS_LOG(DEBUG) << "flID: " << flID.c_str() << ",now time: " << now << ",requestTime: " << requestTime;

  int64_t diff = now - requestTime;
  if (abs(diff) > replayAttackTimeDiff) {
    return false;
  }
  MS_LOG(DEBUG) << "verifyTimeStamp success.";
  return true;
}

void CertVerify::sha256Hash(const std::string &src, uint8_t *hash, const int len) const {
  if (len <= 0) {
    return;
  }
  SHA256_CTX sha_ctx;
  int ret = SHA256_Init(&sha_ctx);
  if (ret != 1) {
    return;
  }
  ret = SHA256_Update(&sha_ctx, src.c_str(), IntToSize(src.size()));
  if (ret != 1) {
    return;
  }
  ret = SHA256_Final(hash, &sha_ctx);
  if (ret != 1) {
    return;
  }
}

bool CertVerify::verifyRSAKey(const std::string &keyAttestation, const uint8_t *srcData, const uint8_t *signData,
                              int srcDataLen) {
  if (keyAttestation.empty() || signData == nullptr || srcData == nullptr || srcDataLen <= 0) {
    MS_LOG(WARNING) << "keyAttestation or signData or srcData is invalid.";
    return false;
  }
  bool result = true;
  X509 *keyAttestationCertObj = nullptr;
  EVP_PKEY *pubKey = nullptr;
  do {
    keyAttestationCertObj = readCertFromPerm(keyAttestation);
    pubKey = X509_get_pubkey(keyAttestationCertObj);
    RSA *pRSAPublicKey = EVP_PKEY_get0_RSA(pubKey);
    if (pRSAPublicKey == nullptr) {
      MS_LOG(WARNING) << "get rsa public key failed.";
      result = false;
      break;
    }

    int pubKeyLen = RSA_size(pRSAPublicKey);
    unsigned char buffer[256];
    int ret = RSA_public_decrypt(pubKeyLen, signData, buffer, pRSAPublicKey, RSA_NO_PADDING);
    if (ret == -1) {
      MS_LOG(WARNING) << "rsa public decrypt failed.";
      result = false;
      break;
    }

    int saltLen = -2;
    ret = RSA_verify_PKCS1_PSS(pRSAPublicKey, srcData, EVP_sha256(), buffer, saltLen);
    if (ret != 1) {
      uint64_t ulErr = ERR_get_error();
      char szErrMsg[1024] = {0};
      MS_LOG(WARNING) << "verify WARNING. WARNING number: " << ulErr;
      std::string str_res = ERR_error_string(ulErr, szErrMsg);
      MS_LOG(WARNING) << szErrMsg;
      if (str_res.empty()) {
        result = false;
        break;
      }
      result = false;
      break;
    }
  } while (0);
  EVP_PKEY_free(pubKey);
  X509_free(keyAttestationCertObj);
  CRYPTO_cleanup_all_ex_data();

  MS_LOG(DEBUG) << "verifyRSAKey end.";
  return result;
}

bool CertVerify::initRootCertAndCRL(const std::string rootFirstCaFilePath, const std::string rootSecondCaFilePath,
                                    const std::string equipCrlPath, const uint64_t replay_attack_time_diff) {
  if (rootFirstCaFilePath.empty() || rootSecondCaFilePath.empty()) {
    MS_LOG(WARNING) << "the root or crl path is empty.";
    return false;
  }

  if (!checkFileExists(rootFirstCaFilePath)) {
    MS_LOG(WARNING) << "The rootFirstCaFilePath is not exist.";
    return false;
  }
  if (!checkFileExists(rootSecondCaFilePath)) {
    MS_LOG(WARNING) << "The rootSecondCaFilePath is not exist.";
    return false;
  }

  if (!checkFileExists(equipCrlPath)) {
    MS_LOG(WARNING) << "The equipCrlPath is not exist.";
  }
  replayAttackTimeDiff = UlongToLong(replay_attack_time_diff);
  return true;
}

bool CertVerify::verifyCertAndSign(const std::string &flID, const std::string &timeStamp, const unsigned char *signData,
                                   const std::string &keyAttestation, const std::string &equipCert,
                                   const std::string &equipCACert, const std::string &rootFirstCAPath,
                                   const std::string &rootSecondCAPath, const std::string &equipCrlPath) {
  if (!verifyEquipCertAndFlID(flID, equipCert)) {
    return false;
  }

  if (!verifyCAChain(keyAttestation, equipCert, equipCACert, rootFirstCAPath, rootSecondCAPath)) {
    return false;
  }

  if (!verifyCRL(equipCert, equipCrlPath)) {
    return false;
  }

  if (!verifyRSAKey(keyAttestation, signData, flID, timeStamp)) {
    return false;
  }

  if (!verifyTimeStamp(flID, timeStamp)) {
    return false;
  }
  return true;
}
#else
bool CertVerify::verifyTimeStamp(const std::string &flID, const std::string &timeStamp) const {
  MS_LOG(WARNING) << "verifyTimeStamp in win32 platform.";
  return false;
}
void CertVerify::sha256Hash(const uint8_t *src, const int src_len, uint8_t *hash, const int len) const {
  MS_LOG(WARNING) << "sha256Hash in win32 platform.";
}
bool CertVerify::verifyRSAKey(const std::string &keyAttestation, const uint8_t *srcData, const uint8_t *signData,
                              int srcDataLen) {
  MS_LOG(WARNING) << "verifyRSAKey in win32 platform.";
  return false;
}
bool CertVerify::initRootCertAndCRL(const std::string rootFirstCaFilePath, const std::string rootSecondCaFilePath,
                                    const std::string equipCrlPath, const uint64_t replay_attack_time_diff) {
  MS_LOG(WARNING) << "initRootCertAndCRL in win32 platform.";
  return false;
}
bool CertVerify::verifyCertAndSign(const std::string &flID, const std::string &timeStamp, const unsigned char *signData,
                                   const std::string &keyAttestation, const std::string &equipCert,
                                   const std::string &equipCACert, const std::string &rootFirstCAPath,
                                   const std::string &rootSecondCAPath, const std::string &equipCrlPath) {
  MS_LOG(WARNING) << "verifyCertAndSign in win32 platform.";
  return false;
}
#endif
}  // namespace server
}  // namespace ps
}  // namespace mindspore
