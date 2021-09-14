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

#ifndef MINDSPORE_KEY_AGREEMENT_H
#define MINDSPORE_KEY_AGREEMENT_H

#ifndef _WIN32
#include <openssl/dh.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#endif
#include "utils/log_adapter.h"

#define KEY_LEN 32
#define SALT_LEN 32
#define ITERATION 10000

namespace mindspore {
namespace armour {

#ifdef _WIN32
class PublicKey {};
class PrivateKey {};
#else
class PublicKey {
 public:
  explicit PublicKey(EVP_PKEY *evpKey);
  ~PublicKey();
  EVP_PKEY *evpPubKey;
};

class PrivateKey {
 public:
  explicit PrivateKey(EVP_PKEY *evpKey);
  ~PrivateKey();
  int Exchange(PublicKey *peerPublicKey, int key_len, const unsigned char *salt, int salt_len,
               unsigned char *exchangeKey);
  int GetPrivateBytes(size_t *len, unsigned char *priKeyBytes) const;
  int GetPublicBytes(size_t *len, unsigned char *pubKeyBytes) const;
  EVP_PKEY *evpPrivKey;
};
#endif

class KeyAgreement {
 public:
  static PrivateKey *GeneratePrivKey();
  static PublicKey *GeneratePubKey(PrivateKey *privKey);
  static PrivateKey *FromPrivateBytes(const unsigned char *data, size_t len);
  static PublicKey *FromPublicBytes(const unsigned char *data, size_t len);
  static int ComputeSharedKey(PrivateKey *privKey, PublicKey *peerPublicKey, int key_len, const unsigned char *salt,
                              int salt_len, unsigned char *exchangeKey);
};

}  // namespace armour
}  // namespace mindspore
#endif  // MINDSPORE_KEY_AGREEMENT_H
