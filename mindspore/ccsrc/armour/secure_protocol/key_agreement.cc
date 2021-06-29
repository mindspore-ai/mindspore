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

#include "armour/secure_protocol/key_agreement.h"

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

PrivateKey *KeyAgreement::FromPrivateBytes(unsigned char *data, int len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

PublicKey *KeyAgreement::FromPublicBytes(unsigned char *data, int len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return NULL;
}

int KeyAgreement::ComputeSharedKey(PrivateKey *privKey, PublicKey *peerPublicKey, int key_len,
                                   const unsigned char *salt, int salt_len, unsigned char *exchangeKey) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
PublicKey::PublicKey(EVP_PKEY *evpKey) { evpPubKey = evpKey; }

PublicKey::~PublicKey() { EVP_PKEY_free(evpPubKey); }

PrivateKey::PrivateKey(EVP_PKEY *evpKey) { evpPrivKey = evpKey; }

PrivateKey::~PrivateKey() { EVP_PKEY_free(evpPrivKey); }

int PrivateKey::GetPrivateBytes(size_t *len, unsigned char *privKeyBytes) {
  if (!EVP_PKEY_get_raw_private_key(evpPrivKey, privKeyBytes, len)) {
    return -1;
  }
  return 0;
}

int PrivateKey::GetPublicBytes(size_t *len, unsigned char *pubKeyBytes) {
  if (!EVP_PKEY_get_raw_public_key(evpPrivKey, pubKeyBytes, len)) {
    return -1;
  }
  return 0;
}

int PrivateKey::Exchange(PublicKey *peerPublicKey, int key_len, const unsigned char *salt, int salt_len,
                         unsigned char *exchangeKey) {
  EVP_PKEY_CTX *ctx;
  size_t len = 0;
  ctx = EVP_PKEY_CTX_new(evpPrivKey, NULL);
  if (!ctx) {
    MS_LOG(ERROR) << "EVP_PKEY_CTX_new failed!";
    return -1;
  }
  if (EVP_PKEY_derive_init(ctx) <= 0) {
    MS_LOG(ERROR) << "EVP_PKEY_derive_init failed!";
    return -1;
  }
  if (EVP_PKEY_derive_set_peer(ctx, peerPublicKey->evpPubKey) <= 0) {
    MS_LOG(ERROR) << "EVP_PKEY_derive_set_peer failed!";
    return -1;
  }
  unsigned char *secret;
  if (EVP_PKEY_derive(ctx, NULL, &len) <= 0) {
    MS_LOG(ERROR) << "get derive key size failed!";
    return -1;
  }

  secret = (unsigned char *)OPENSSL_malloc(len);
  if (!secret) {
    MS_LOG(ERROR) << "malloc secret memory failed!";
    return -1;
  }

  if (EVP_PKEY_derive(ctx, secret, &len) <= 0) {
    MS_LOG(ERROR) << "derive key failed!";
    return -1;
  }

  if (!PKCS5_PBKDF2_HMAC((char *)secret, len, salt, salt_len, ITERATION, EVP_sha256(), key_len, exchangeKey)) {
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
  if (!pctx) {
    return NULL;
  }
  if (EVP_PKEY_keygen_init(pctx) <= 0) {
    return NULL;
  }
  if (EVP_PKEY_keygen(pctx, &evpKey) <= 0) {
    return NULL;
  }
  EVP_PKEY_CTX_free(pctx);
  PrivateKey *privKey = new PrivateKey(evpKey);
  return privKey;
}

PublicKey *KeyAgreement::GeneratePubKey(PrivateKey *privKey) {
  unsigned char *pubKeyBytes;
  size_t len = 0;
  if (!EVP_PKEY_get_raw_public_key(privKey->evpPrivKey, NULL, &len)) {
    return NULL;
  }
  pubKeyBytes = (unsigned char *)OPENSSL_malloc(len);
  if (!EVP_PKEY_get_raw_public_key(privKey->evpPrivKey, pubKeyBytes, &len)) {
    return NULL;
  }
  EVP_PKEY *evp_pubKey = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, (unsigned char *)pubKeyBytes, len);
  OPENSSL_free(pubKeyBytes);
  PublicKey *pubKey = new PublicKey(evp_pubKey);
  return pubKey;
}

PrivateKey *KeyAgreement::FromPrivateBytes(unsigned char *data, int len) {
  EVP_PKEY *evp_Key = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, NULL, data, len);
  if (evp_Key == NULL) {
    return NULL;
  }
  PrivateKey *privKey = new PrivateKey(evp_Key);
  return privKey;
}

PublicKey *KeyAgreement::FromPublicBytes(unsigned char *data, int len) {
  EVP_PKEY *evp_pubKey = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, data, len);
  if (evp_pubKey == NULL) {
    MS_LOG(ERROR) << "create evp_pubKey from raw bytes fail";
    return NULL;
  }
  PublicKey *pubKey = new PublicKey(evp_pubKey);
  return pubKey;
}

int KeyAgreement::ComputeSharedKey(PrivateKey *privKey, PublicKey *peerPublicKey, int key_len,
                                   const unsigned char *salt, int salt_len, unsigned char *exchangeKey) {
  return privKey->Exchange(peerPublicKey, key_len, salt, salt_len, exchangeKey);
}
#endif

}  // namespace armour
}  // namespace mindspore
