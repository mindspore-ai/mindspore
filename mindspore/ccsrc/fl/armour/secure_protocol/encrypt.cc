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

#include "fl/armour/secure_protocol/encrypt.h"

namespace mindspore {
namespace armour {
AESEncrypt::AESEncrypt(const uint8_t *key, int key_len, const uint8_t *ivec, int ivec_len, const AES_MODE mode) {
  priv_key_ = key;
  priv_key_len_ = key_len;
  ivec_ = ivec;
  ivec_len_ = ivec_len;
  aes_mode_ = mode;
}

AESEncrypt::~AESEncrypt() {}

#if defined(_WIN32)
int AESEncrypt::EncryptData(const uint8_t *data, const int len, uint8_t *encrypt_data, int *encrypt_len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

int AESEncrypt::DecryptData(const uint8_t *encrypt_data, const int encrypt_len, uint8_t *data, int *len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
int AESEncrypt::EncryptData(const uint8_t *data, const int len, uint8_t *encrypt_data, int *encrypt_len) const {
  int ret;
  if (priv_key_ == nullptr || ivec_ == nullptr) {
    MS_LOG(ERROR) << "private key or init vector is invalid.";
    return -1;
  }
  if (priv_key_len_ != KEY_LENGTH_16 && priv_key_len_ != KEY_LENGTH_32) {
    MS_LOG(ERROR) << "key length is invalid.";
    return -1;
  }
  if (ivec_len_ != AES_IV_SIZE) {
    MS_LOG(ERROR) << "initial vector size is invalid.";
    return -1;
  }
  if (data == nullptr || len <= 0 || encrypt_data == nullptr || encrypt_len == nullptr) {
    MS_LOG(ERROR) << "input data is invalid.";
    return -1;
  }
  ret = evp_aes_encrypt(data, len, priv_key_, ivec_, encrypt_data, encrypt_len);
  if (ret != 0) {
    return -1;
  }
  return 0;
}

int AESEncrypt::DecryptData(const uint8_t *encrypt_data, const int encrypt_len, uint8_t *data, int *len) const {
  int ret = 0;
  if (priv_key_ == nullptr || ivec_ == nullptr) {
    MS_LOG(ERROR) << "private key or init vector is invalid.";
    return -1;
  }
  if (priv_key_len_ != KEY_LENGTH_16 && priv_key_len_ != KEY_LENGTH_32) {
    MS_LOG(ERROR) << "key length is invalid.";
    return -1;
  }
  if (ivec_len_ != AES_IV_SIZE) {
    MS_LOG(ERROR) << "initial vector size is invalid.";
    return -1;
  }
  if (data == nullptr || encrypt_len <= 0 || encrypt_data == nullptr || len == nullptr) {
    MS_LOG(ERROR) << "input data is invalid.";
    return -1;
  }
  if (aes_mode_ == AES_CBC || aes_mode_ == AES_CTR) {
    ret = evp_aes_decrypt(encrypt_data, encrypt_len, priv_key_, ivec_, data, len);
  } else {
    MS_LOG(ERROR) << "This encryption mode is not supported!";
  }
  if (ret != 1) {
    return -1;
  }
  return 0;
}

int AESEncrypt::evp_aes_encrypt(const uint8_t *data, const int len, const uint8_t *key, const uint8_t *ivec,
                                uint8_t *encrypt_data, int *encrypt_len) const {
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (ctx == NULL) {
    return -1;
  }
  int out_len;
  int ret;
  if (aes_mode_ == AES_CBC) {
    switch (priv_key_len_) {
      case KEY_LENGTH_16:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, ivec);
        break;
      case KEY_LENGTH_32:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      EVP_CIPHER_CTX_free(ctx);
      return -1;
    }
    EVP_CIPHER_CTX_set_padding(ctx, EVP_PADDING_PKCS7);
  } else if (aes_mode_ == AES_CTR) {
    switch (priv_key_len_) {
      case KEY_LENGTH_16:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, ivec);
        break;
      case KEY_LENGTH_32:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      EVP_CIPHER_CTX_free(ctx);
      return -1;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported encryption mode";
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  ret = EVP_EncryptUpdate(ctx, encrypt_data, &out_len, data, len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  *encrypt_len = out_len;
  ret = EVP_EncryptFinal_ex(ctx, encrypt_data + *encrypt_len, &out_len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  *encrypt_len += out_len;
  EVP_CIPHER_CTX_free(ctx);
  return 0;
}

int AESEncrypt::evp_aes_decrypt(const uint8_t *encrypt_data, const int len, const uint8_t *key, const uint8_t *ivec,
                                uint8_t *decrypt_data, int *decrypt_len) const {
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (ctx == NULL) {
    return -1;
  }
  int out_len;
  int ret;
  if (aes_mode_ == AES_CBC) {
    switch (priv_key_len_) {
      case KEY_LENGTH_16:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, ivec);
        break;
      case KEY_LENGTH_32:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      EVP_CIPHER_CTX_free(ctx);
      return -1;
    }
  } else if (aes_mode_ == AES_CTR) {
    switch (priv_key_len_) {
      case KEY_LENGTH_16:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, ivec);
        break;
      case KEY_LENGTH_32:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      EVP_CIPHER_CTX_free(ctx);
      return -1;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported encryption mode";
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }

  ret = EVP_DecryptUpdate(ctx, decrypt_data, &out_len, encrypt_data, len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  *decrypt_len = out_len;
  ret = EVP_DecryptFinal_ex(ctx, decrypt_data + *decrypt_len, &out_len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  *decrypt_len += out_len;
  EVP_CIPHER_CTX_free(ctx);
  return 0;
}
#endif
}  // namespace armour
}  // namespace mindspore
