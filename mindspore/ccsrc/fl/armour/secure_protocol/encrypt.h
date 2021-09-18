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

#ifndef MINDSPORE_ARMOUR_ENCRYPT_H
#define MINDSPORE_ARMOUR_ENCRYPT_H

#ifndef _WIN32
#include <openssl/evp.h>
#endif
#include "utils/log_adapter.h"

#define AES_IV_SIZE 16
#define KEY_LENGTH_32 32
#define KEY_LENGTH_16 16

namespace mindspore {
namespace armour {
class Encrypt {};
enum AES_MODE {
  AES_CBC = 0,
  AES_CTR = 1,
};
class SymmetricEncrypt : Encrypt {};

class AESEncrypt : SymmetricEncrypt {
 public:
  AESEncrypt(const uint8_t *key, int key_len, const uint8_t *ivec, int ivec_len, AES_MODE mode);
  ~AESEncrypt();
  int EncryptData(const uint8_t *data, const int len, uint8_t *encrypt_data, int *encrypt_len) const;
  int DecryptData(const uint8_t *encrypt_data, const int encrypt_len, uint8_t *data, int *len) const;

 private:
  const uint8_t *priv_key_;
  int priv_key_len_;
  const uint8_t *ivec_;
  int ivec_len_;
  AES_MODE aes_mode_;
  int evp_aes_encrypt(const uint8_t *data, const int len, const uint8_t *key, const uint8_t *ivec,
                      uint8_t *encrypt_data, int *encrypt_len) const;
  int evp_aes_decrypt(const uint8_t *encrypt_data, const int len, const uint8_t *key, const uint8_t *ivec,
                      uint8_t *decrypt_data, int *decrypt_len) const;
};

}  // namespace armour
}  // namespace mindspore
#endif  // MINDSPORE_ARMOUR_ENCRYPT_H
