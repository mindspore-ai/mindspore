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

#include "fl/armour/secure_protocol/key_agreement.h"

namespace mindspore {
namespace armour {
#ifdef _WIN32
PrivateKey *KeyAgreement::GeneratePrivKey() {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

PublicKey *KeyAgreement::GeneratePubKey(PrivateKey *privKey) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

PrivateKey *KeyAgreement::FromPrivateBytes(uint8_t *data, int len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

PublicKey *KeyAgreement::FromPublicBytes(uint8_t *data, int len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

int KeyAgreement::ComputeSharedKey(PrivateKey *privKey, PublicKey *peerPublicKey, int key_len, const uint8_t *salt,
                                   int salt_len, uint8_t *exchangeKey) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
PublicKey::PublicKey(EVP_PKEY *evpKey) { evpPubKey = evpKey; }

PublicKey::~PublicKey() { EVP_PKEY_free(evpPubKey); }

PrivateKey::PrivateKey(EVP_PKEY *evpKey) { evpPrivKey = evpKey; }

PrivateKey::~PrivateKey() { EVP_PKEY_free(evpPrivKey); }

int PrivateKey::GetPrivateBytes(size_t *len, uint8_t *privKeyBytes) const {
  if (privKeyBytes == nullptr || len == nullptr || evpPrivKey == nullptr) {
    MS_LOG(ERROR) << "input data invalid.";
    return -1;
  }
  if (!EVP_PKEY_get_raw_private_key(evpPrivKey, privKeyBytes, len)) {
    return -1;
  }
  return 0;
}

int PrivateKey::GetPublicBytes(size_t *len, uint8_t *pubKeyBytes) const {
  if (pubKeyBytes == nullptr || len == nullptr || evpPrivKey == nullptr) {
    MS_LOG(ERROR) << "input pubKeyBytes invalid.";
    return -1;
  }
  if (!EVP_PKEY_get_raw_public_key(evpPrivKey, pubKeyBytes, len)) {
    return -1;
  }
  return 0;
}

int PrivateKey::Exchange(PublicKey *peerPublicKey, int key_len, const unsigned char *salt, int salt_len,
                         uint8_t *exchangeKey) {
  if (peerPublicKey == nullptr) {
    MS_LOG(ERROR) << "peerPublicKey is nullptr.";
    return -1;
  }
  if (key_len != KEY_LEN || exchangeKey == nullptr) {
    MS_LOG(ERROR) << "exchangeKey is nullptr or input key_len is incorrect.";
    return -1;
  }
  if (salt == nullptr || salt_len != SALT_LEN) {
    MS_LOG(ERROR) << "input salt in invalid.";
    return -1;
  }
  size_t len = 0;
  EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(evpPrivKey, NULL);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "new context failed!";
    return -1;
  }
  if (EVP_PKEY_derive_init(ctx) <= 0) {
    MS_LOG(ERROR) << "EVP_PKEY_derive_init failed!";
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  if (EVP_PKEY_derive_set_peer(ctx, peerPublicKey->evpPubKey) <= 0) {
    MS_LOG(ERROR) << "EVP_PKEY_derive_set_peer failed!";
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  if (EVP_PKEY_derive(ctx, NULL, &len) <= 0) {
    MS_LOG(ERROR) << "get derive key size failed!";
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  if (len == 0) {
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  uint8_t *secret = reinterpret_cast<uint8_t *>(OPENSSL_malloc(len));
  if (secret == nullptr) {
    MS_LOG(ERROR) << "malloc secret memory failed!";
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }

  if (EVP_PKEY_derive(ctx, secret, &len) <= 0) {
    MS_LOG(ERROR) << "derive key failed!";
    OPENSSL_free(secret);
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  if (!PKCS5_PBKDF2_HMAC(reinterpret_cast<char *>(secret), len, salt, salt_len, ITERATION, EVP_sha256(), key_len,
                         exchangeKey)) {
    OPENSSL_free(secret);
    EVP_PKEY_CTX_free(ctx);
    return -1;
  }
  OPENSSL_free(secret);
  EVP_PKEY_CTX_free(ctx);
  return 0;
}

// using x25519 curve
PrivateKey *KeyAgreement::GeneratePrivKey() {
  EVP_PKEY *evpKey = NULL;
  EVP_PKEY_CTX *pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_X25519, NULL);
  if (pctx == nullptr) {
    return NULL;
  }
  if (EVP_PKEY_keygen_init(pctx) <= 0) {
    EVP_PKEY_CTX_free(pctx);
    return NULL;
  }
  if (EVP_PKEY_keygen(pctx, &evpKey) <= 0) {
    EVP_PKEY_CTX_free(pctx);
    return NULL;
  }
  EVP_PKEY_CTX_free(pctx);
  PrivateKey *privKey = new PrivateKey(evpKey);
  return privKey;
}

PublicKey *KeyAgreement::GeneratePubKey(PrivateKey *privKey) {
  uint8_t *pubKeyBytes;
  size_t len = 0;
  if (privKey == nullptr) {
    return NULL;
  }
  if (!EVP_PKEY_get_raw_public_key(privKey->evpPrivKey, NULL, &len)) {
    return NULL;
  }
  pubKeyBytes = reinterpret_cast<uint8_t *>(OPENSSL_malloc(len));
  if (pubKeyBytes == nullptr) {
    MS_LOG(ERROR) << "malloc secret memory failed!";
    return NULL;
  }

  if (!EVP_PKEY_get_raw_public_key(privKey->evpPrivKey, pubKeyBytes, &len)) {
    MS_LOG(ERROR) << "EVP_PKEY_get_raw_public_key failed!";
    OPENSSL_free(pubKeyBytes);
    return NULL;
  }
  EVP_PKEY *evp_pubKey =
    EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, reinterpret_cast<uint8_t *>(pubKeyBytes), len);
  if (evp_pubKey == NULL) {
    MS_LOG(ERROR) << "EVP_PKEY_new_raw_public_key failed!";
    OPENSSL_free(pubKeyBytes);
    return NULL;
  }
  OPENSSL_free(pubKeyBytes);
  PublicKey *pubKey = new PublicKey(evp_pubKey);
  return pubKey;
}

PrivateKey *KeyAgreement::FromPrivateBytes(const uint8_t *data, size_t len) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "input data is null!";
    return NULL;
  }
  EVP_PKEY *evp_Key = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, NULL, data, len);
  if (evp_Key == NULL) {
    MS_LOG(ERROR) << "create evp_Key from raw bytes failed!";
    return NULL;
  }
  PrivateKey *privKey = new PrivateKey(evp_Key);
  return privKey;
}

PublicKey *KeyAgreement::FromPublicBytes(const uint8_t *data, size_t len) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "input data is null!";
    return NULL;
  }
  EVP_PKEY *evp_pubKey = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, data, len);
  if (evp_pubKey == NULL) {
    MS_LOG(ERROR) << "create evp_pubKey from raw bytes fail";
    return NULL;
  }
  PublicKey *pubKey = new PublicKey(evp_pubKey);
  return pubKey;
}

int KeyAgreement::ComputeSharedKey(PrivateKey *privKey, PublicKey *peerPublicKey, int key_len,
                                   const unsigned char *salt, int salt_len, uint8_t *exchangeKey) {
  if (privKey == nullptr) {
    MS_LOG(ERROR) << "privKey is nullptr!";
    return -1;
  }
  return privKey->Exchange(peerPublicKey, key_len, salt, salt_len, exchangeKey);
}
#endif
}  // namespace armour
}  // namespace mindspore
