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

#define KEY_STEP_MAX 32
#define KEY_STEP_MIN 16
#define PAD_SIZE 5

AESEncrypt::AESEncrypt(const unsigned char *key, int key_len, unsigned char *ivec, int ivec_len, const AES_MODE mode) {
  privKey = key;
  privKeyLen = key_len;
  iVec = ivec;
  iVecLen = ivec_len;
  aesMode = mode;
}

AESEncrypt::~AESEncrypt() {}

#if defined(_WIN32)
int AESEncrypt::EncryptData(const unsigned char *data, const int len, unsigned char *encrypt_data, int *encrypt_len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

int AESEncrypt::DecryptData(const unsigned char *encrypt_data, const int encrypt_len, unsigned char *data, int *len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
int AESEncrypt::EncryptData(const unsigned char *data, const int len, unsigned char *encrypt_data, int *encrypt_len) {
  int ret;
  if (privKeyLen != KEY_STEP_MIN && privKeyLen != KEY_STEP_MAX) {
    MS_LOG(ERROR) << "key length must be 16 or 32!";
    return -1;
  }
  if (iVecLen != INIT_VEC_SIZE) {
    MS_LOG(ERROR) << "initial vector size must be 16!";
    return -1;
  }
  if (aesMode == AES_CBC || aesMode == AES_CTR) {
    ret = evp_aes_encrypt(data, len, privKey, iVec, encrypt_data, encrypt_len);
  } else {
    MS_LOG(ERROR) << "Please use CBC mode or CTR mode, the other modes are not supported!";
    ret = -1;
  }
  if (ret != 0) {
    return -1;
  }
  return 0;
}

int AESEncrypt::DecryptData(const unsigned char *encrypt_data, const int encrypt_len, unsigned char *data, int *len) {
  int ret = 0;
  if (privKeyLen != KEY_STEP_MIN && privKeyLen != KEY_STEP_MAX) {
    MS_LOG(ERROR) << "key length must be 16 or 32!";
    return -1;
  }
  if (iVecLen != INIT_VEC_SIZE) {
    MS_LOG(ERROR) << "initial vector size must be 16!";
    return -1;
  }
  if (aesMode == AES_CBC || aesMode == AES_CTR) {
    ret = evp_aes_decrypt(encrypt_data, encrypt_len, privKey, iVec, data, len);
  } else {
    MS_LOG(ERROR) << "Please use CBC mode or CTR mode, the other modes are not supported!";
  }
  if (ret != 1) {
    return -1;
  }
  return 0;
}

int AESEncrypt::evp_aes_encrypt(const unsigned char *data, const int len, const unsigned char *key, unsigned char *ivec,
                                unsigned char *encrypt_data, int *encrypt_len) {
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  int out_len;
  int ret = 0;
  if (aesMode == AES_CBC) {
    switch (privKeyLen) {
      case 16:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, ivec);
        break;
      case 32:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_EncryptInit_ex CBC fail!";
      return -1;
    }
    EVP_CIPHER_CTX_set_key_length(ctx, EVP_MAX_KEY_LENGTH);
    EVP_CIPHER_CTX_set_padding(ctx, PAD_SIZE);
  } else if (aesMode == AES_CTR) {
    switch (privKeyLen) {
      case 16:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, ivec);
        break;
      case 32:
        ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_EncryptInit_ex CTR fail!";
      return -1;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported AES mode";
    return -1;
  }
  ret = EVP_EncryptUpdate(ctx, encrypt_data, &out_len, data, len);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptUpdate fail!";
    return -1;
  }
  *encrypt_len = out_len;
  ret = EVP_EncryptFinal_ex(ctx, encrypt_data + *encrypt_len, &out_len);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptFinal_ex fail!";
    return -1;
  }
  *encrypt_len += out_len;
  EVP_CIPHER_CTX_free(ctx);
  return 0;
}

int AESEncrypt::evp_aes_decrypt(const unsigned char *encrypt_data, const int len, const unsigned char *key,
                                unsigned char *ivec, unsigned char *decrypt_data, int *decrypt_len) {
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  int out_len;
  int ret = 0;
  if (aesMode == AES_CBC) {
    switch (privKeyLen) {
      case 16:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, ivec);
        break;
      case 32:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
    if (ret != 1) {
      return -1;
    }
    EVP_CIPHER_CTX_set_key_length(ctx, EVP_MAX_KEY_LENGTH);
  } else if (aesMode == AES_CTR) {
    switch (privKeyLen) {
      case 16:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, ivec);
        break;
      case 32:
        ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, ivec);
        break;
      default:
        MS_LOG(ERROR) << "key length is incorrect!";
        ret = -1;
    }
  } else {
    ret = -1;
  }

  if (ret != 1) {
    return -1;
  }
  ret = EVP_DecryptUpdate(ctx, decrypt_data, &out_len, encrypt_data, len);
  if (ret != 1) {
    return -1;
  }
  *decrypt_len = out_len;
  ret = EVP_DecryptFinal_ex(ctx, decrypt_data + *decrypt_len, &out_len);
  if (ret != 1) {
    return -1;
  }
  *decrypt_len += out_len;
  EVP_CIPHER_CTX_free(ctx);

  return 0;
}
#endif

}  // namespace armour
}  // namespace mindspore
